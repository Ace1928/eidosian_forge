import os
import copy
import enum
from operator import ior
import logging
from fontTools.colorLib.builder import MAX_PAINT_COLR_LAYER_COUNT, LayerReuseCache
from fontTools.misc import classifyTools
from fontTools.misc.roundTools import otRound
from fontTools.misc.treeTools import build_n_ary_tree
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables import otBase as otBase
from fontTools.ttLib.tables.otConverters import BaseFixedValue
from fontTools.ttLib.tables.otTraverse import dfs_base_table
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.varLib import builder, models, varStore
from fontTools.varLib.models import nonNone, allNone, allEqual, allEqualTo, subList
from fontTools.varLib.varStore import VarStoreInstancer
from functools import reduce
from fontTools.otlLib.builder import buildSinglePos
from fontTools.otlLib.optimize.gpos import (
from .errors import (
class COLRVariationMerger(VariationMerger):
    """A specialized VariationMerger that takes multiple master fonts containing
    COLRv1 tables, and builds a variable COLR font.

    COLR tables are special in that variable subtables can be associated with
    multiple delta-set indices (via VarIndexBase).
    They also contain tables that must change their type (not simply the Format)
    as they become variable (e.g. Affine2x3 -> VarAffine2x3) so this merger takes
    care of that too.
    """

    def __init__(self, model, axisTags, font, allowLayerReuse=True):
        VariationMerger.__init__(self, model, axisTags, font)
        self.varIndexCache = {}
        self.varIdxes = []
        self.varTableIds = set()
        self.layers = []
        self.layerReuseCache = None
        if allowLayerReuse:
            self.layerReuseCache = LayerReuseCache()
        self._doneBaseGlyphs = False

    def mergeTables(self, font, master_ttfs, tableTags=('COLR',)):
        if 'COLR' in tableTags and 'COLR' in font:
            self.expandPaintColrLayers(font['COLR'].table)
        VariationMerger.mergeTables(self, font, master_ttfs, tableTags)

    def checkFormatEnum(self, out, lst, validate=lambda _: True):
        fmt = out.Format
        formatEnum = out.formatEnum
        ok = False
        try:
            fmt = formatEnum(fmt)
        except ValueError:
            pass
        else:
            ok = validate(fmt)
        if not ok:
            raise UnsupportedFormat(self, subtable=type(out).__name__, value=fmt)
        expected = fmt
        got = []
        for v in lst:
            fmt = getattr(v, 'Format', None)
            try:
                fmt = formatEnum(fmt)
            except ValueError:
                pass
            got.append(fmt)
        if not allEqualTo(expected, got):
            raise InconsistentFormats(self, subtable=type(out).__name__, expected=expected, got=got)
        return expected

    def mergeSparseDict(self, out, lst):
        for k in out.keys():
            try:
                self.mergeThings(out[k], [v.get(k) for v in lst])
            except VarLibMergeError as e:
                e.stack.append(f'[{k!r}]')
                raise

    def mergeAttrs(self, out, lst, attrs):
        for attr in attrs:
            value = getattr(out, attr)
            values = [getattr(item, attr) for item in lst]
            try:
                self.mergeThings(value, values)
            except VarLibMergeError as e:
                e.stack.append(f'.{attr}')
                raise

    def storeMastersForAttr(self, out, lst, attr):
        master_values = [getattr(item, attr) for item in lst]
        is_fixed_size_float = False
        conv = out.getConverterByName(attr)
        if isinstance(conv, BaseFixedValue):
            is_fixed_size_float = True
            master_values = [conv.toInt(v) for v in master_values]
        baseValue = master_values[0]
        varIdx = ot.NO_VARIATION_INDEX
        if not allEqual(master_values):
            baseValue, varIdx = self.store_builder.storeMasters(master_values)
        if is_fixed_size_float:
            baseValue = conv.fromInt(baseValue)
        return (baseValue, varIdx)

    def storeVariationIndices(self, varIdxes) -> int:
        key = tuple(varIdxes)
        varIndexBase = self.varIndexCache.get(key)
        if varIndexBase is None:
            for i in range(len(self.varIdxes) - len(varIdxes) + 1):
                if self.varIdxes[i:i + len(varIdxes)] == varIdxes:
                    self.varIndexCache[key] = varIndexBase = i
                    break
        if varIndexBase is None:
            for n in range(len(varIdxes) - 1, 0, -1):
                if self.varIdxes[-n:] == varIdxes[:n]:
                    varIndexBase = len(self.varIdxes) - n
                    self.varIndexCache[key] = varIndexBase
                    self.varIdxes.extend(varIdxes[n:])
                    break
        if varIndexBase is None:
            self.varIndexCache[key] = varIndexBase = len(self.varIdxes)
            self.varIdxes.extend(varIdxes)
        return varIndexBase

    def mergeVariableAttrs(self, out, lst, attrs) -> int:
        varIndexBase = ot.NO_VARIATION_INDEX
        varIdxes = []
        for attr in attrs:
            baseValue, varIdx = self.storeMastersForAttr(out, lst, attr)
            setattr(out, attr, baseValue)
            varIdxes.append(varIdx)
        if any((v != ot.NO_VARIATION_INDEX for v in varIdxes)):
            varIndexBase = self.storeVariationIndices(varIdxes)
        return varIndexBase

    @classmethod
    def convertSubTablesToVarType(cls, table):
        for path in dfs_base_table(table, skip_root=True, predicate=lambda path: getattr(type(path[-1].value), 'VarType', None) is not None):
            st = path[-1]
            subTable = st.value
            varType = type(subTable).VarType
            newSubTable = varType()
            newSubTable.__dict__.update(subTable.__dict__)
            newSubTable.populateDefaults()
            parent = path[-2].value
            if st.index is not None:
                getattr(parent, st.name)[st.index] = newSubTable
            else:
                setattr(parent, st.name, newSubTable)

    @staticmethod
    def expandPaintColrLayers(colr):
        """Rebuild LayerList without PaintColrLayers reuse.

        Each base paint graph is fully DFS-traversed (with exception of PaintColrGlyph
        which are irrelevant for this); any layers referenced via PaintColrLayers are
        collected into a new LayerList and duplicated when reuse is detected, to ensure
        that all paints are distinct objects at the end of the process.
        PaintColrLayers's FirstLayerIndex/NumLayers are updated so that no overlap
        is left. Also, any consecutively nested PaintColrLayers are flattened.
        The COLR table's LayerList is replaced with the new unique layers.
        A side effect is also that any layer from the old LayerList which is not
        referenced by any PaintColrLayers is dropped.
        """
        if not colr.LayerList:
            return
        uniqueLayerIDs = set()
        newLayerList = []
        for rec in colr.BaseGlyphList.BaseGlyphPaintRecord:
            frontier = [rec.Paint]
            while frontier:
                paint = frontier.pop()
                if paint.Format == ot.PaintFormat.PaintColrGlyph:
                    continue
                elif paint.Format == ot.PaintFormat.PaintColrLayers:
                    children = list(_flatten_layers(paint, colr))
                    first_layer_index = len(newLayerList)
                    for layer in children:
                        if id(layer) in uniqueLayerIDs:
                            layer = copy.deepcopy(layer)
                            assert id(layer) not in uniqueLayerIDs
                        newLayerList.append(layer)
                        uniqueLayerIDs.add(id(layer))
                    paint.FirstLayerIndex = first_layer_index
                    paint.NumLayers = len(children)
                else:
                    children = paint.getChildren(colr)
                frontier.extend(reversed(children))
        assert len(newLayerList) == len(uniqueLayerIDs)
        colr.LayerList.Paint = newLayerList
        colr.LayerList.LayerCount = len(newLayerList)