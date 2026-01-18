from collections import namedtuple
from fontTools.misc import sstruct
from fontTools import ttLib
from fontTools import version
from fontTools.misc.transform import DecomposedTransform
from fontTools.misc.textTools import tostr, safeEval, pad
from fontTools.misc.arrayTools import updateBounds, pointInRect
from fontTools.misc.bezierTools import calcQuadraticBounds
from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.vector import Vector
from numbers import Number
from . import DefaultTable
from . import ttProgram
import sys
import struct
import array
import logging
import math
import os
from fontTools.misc import xmlWriter
from fontTools.misc.filenames import userNameToFileName
from fontTools.misc.loggingTools import deprecateFunction
from enum import IntFlag
from functools import partial
from types import SimpleNamespace
from typing import Set
class table__g_l_y_f(DefaultTable.DefaultTable):
    """Glyph Data Table

    This class represents the `glyf <https://docs.microsoft.com/en-us/typography/opentype/spec/glyf>`_
    table, which contains outlines for glyphs in TrueType format. In many cases,
    it is easier to access and manipulate glyph outlines through the ``GlyphSet``
    object returned from :py:meth:`fontTools.ttLib.ttFont.getGlyphSet`::

                    >> from fontTools.pens.boundsPen import BoundsPen
                    >> glyphset = font.getGlyphSet()
                    >> bp = BoundsPen(glyphset)
                    >> glyphset["A"].draw(bp)
                    >> bp.bounds
                    (19, 0, 633, 716)

    However, this class can be used for low-level access to the ``glyf`` table data.
    Objects of this class support dictionary-like access, mapping glyph names to
    :py:class:`Glyph` objects::

                    >> glyf = font["glyf"]
                    >> len(glyf["Aacute"].components)
                    2

    Note that when adding glyphs to the font via low-level access to the ``glyf``
    table, the new glyphs must also be added to the ``hmtx``/``vmtx`` table::

                    >> font["glyf"]["divisionslash"] = Glyph()
                    >> font["hmtx"]["divisionslash"] = (640, 0)

    """
    dependencies = ['fvar']
    padding = 1

    def decompile(self, data, ttFont):
        self.axisTags = [axis.axisTag for axis in ttFont['fvar'].axes] if 'fvar' in ttFont else []
        loca = ttFont['loca']
        pos = int(loca[0])
        nextPos = 0
        noname = 0
        self.glyphs = {}
        self.glyphOrder = glyphOrder = ttFont.getGlyphOrder()
        self._reverseGlyphOrder = {}
        for i in range(0, len(loca) - 1):
            try:
                glyphName = glyphOrder[i]
            except IndexError:
                noname = noname + 1
                glyphName = 'ttxautoglyph%s' % i
            nextPos = int(loca[i + 1])
            glyphdata = data[pos:nextPos]
            if len(glyphdata) != nextPos - pos:
                raise ttLib.TTLibError("not enough 'glyf' table data")
            glyph = Glyph(glyphdata)
            self.glyphs[glyphName] = glyph
            pos = nextPos
        if len(data) - nextPos >= 4:
            log.warning("too much 'glyf' table data: expected %d, received %d bytes", nextPos, len(data))
        if noname:
            log.warning('%s glyphs have no name', noname)
        if ttFont.lazy is False:
            self.ensureDecompiled()

    def ensureDecompiled(self, recurse=False):
        for glyph in self.glyphs.values():
            glyph.expand(self)

    def compile(self, ttFont):
        self.axisTags = [axis.axisTag for axis in ttFont['fvar'].axes] if 'fvar' in ttFont else []
        if not hasattr(self, 'glyphOrder'):
            self.glyphOrder = ttFont.getGlyphOrder()
        padding = self.padding
        assert padding in (0, 1, 2, 4)
        locations = []
        currentLocation = 0
        dataList = []
        recalcBBoxes = ttFont.recalcBBoxes
        boundsDone = set()
        for glyphName in self.glyphOrder:
            glyph = self.glyphs[glyphName]
            glyphData = glyph.compile(self, recalcBBoxes, boundsDone=boundsDone)
            if padding > 1:
                glyphData = pad(glyphData, size=padding)
            locations.append(currentLocation)
            currentLocation = currentLocation + len(glyphData)
            dataList.append(glyphData)
        locations.append(currentLocation)
        if padding == 1 and currentLocation < 131072:
            indices = [i for i, glyphData in enumerate(dataList) if len(glyphData) % 2 == 1]
            if indices and currentLocation + len(indices) < 131072:
                for i in indices:
                    dataList[i] += b'\x00'
                currentLocation = 0
                for i, glyphData in enumerate(dataList):
                    locations[i] = currentLocation
                    currentLocation += len(glyphData)
                locations[len(dataList)] = currentLocation
        data = b''.join(dataList)
        if 'loca' in ttFont:
            ttFont['loca'].set(locations)
        if 'maxp' in ttFont:
            ttFont['maxp'].numGlyphs = len(self.glyphs)
        if not data:
            data = b'\x00'
        return data

    def toXML(self, writer, ttFont, splitGlyphs=False):
        notice = 'The xMin, yMin, xMax and yMax values\nwill be recalculated by the compiler.'
        glyphNames = ttFont.getGlyphNames()
        if not splitGlyphs:
            writer.newline()
            writer.comment(notice)
            writer.newline()
            writer.newline()
        numGlyphs = len(glyphNames)
        if splitGlyphs:
            path, ext = os.path.splitext(writer.file.name)
            existingGlyphFiles = set()
        for glyphName in glyphNames:
            glyph = self.get(glyphName)
            if glyph is None:
                log.warning("glyph '%s' does not exist in glyf table", glyphName)
                continue
            if glyph.numberOfContours:
                if splitGlyphs:
                    glyphPath = userNameToFileName(tostr(glyphName, 'utf-8'), existingGlyphFiles, prefix=path + '.', suffix=ext)
                    existingGlyphFiles.add(glyphPath.lower())
                    glyphWriter = xmlWriter.XMLWriter(glyphPath, idlefunc=writer.idlefunc, newlinestr=writer.newlinestr)
                    glyphWriter.begintag('ttFont', ttLibVersion=version)
                    glyphWriter.newline()
                    glyphWriter.begintag('glyf')
                    glyphWriter.newline()
                    glyphWriter.comment(notice)
                    glyphWriter.newline()
                    writer.simpletag('TTGlyph', src=os.path.basename(glyphPath))
                else:
                    glyphWriter = writer
                glyphWriter.begintag('TTGlyph', [('name', glyphName), ('xMin', glyph.xMin), ('yMin', glyph.yMin), ('xMax', glyph.xMax), ('yMax', glyph.yMax)])
                glyphWriter.newline()
                glyph.toXML(glyphWriter, ttFont)
                glyphWriter.endtag('TTGlyph')
                glyphWriter.newline()
                if splitGlyphs:
                    glyphWriter.endtag('glyf')
                    glyphWriter.newline()
                    glyphWriter.endtag('ttFont')
                    glyphWriter.newline()
                    glyphWriter.close()
            else:
                writer.simpletag('TTGlyph', name=glyphName)
                writer.comment('contains no outline data')
                if not splitGlyphs:
                    writer.newline()
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name != 'TTGlyph':
            return
        if not hasattr(self, 'glyphs'):
            self.glyphs = {}
        if not hasattr(self, 'glyphOrder'):
            self.glyphOrder = ttFont.getGlyphOrder()
        glyphName = attrs['name']
        log.debug("unpacking glyph '%s'", glyphName)
        glyph = Glyph()
        for attr in ['xMin', 'yMin', 'xMax', 'yMax']:
            setattr(glyph, attr, safeEval(attrs.get(attr, '0')))
        self.glyphs[glyphName] = glyph
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            glyph.fromXML(name, attrs, content, ttFont)
        if not ttFont.recalcBBoxes:
            glyph.compact(self, 0)

    def setGlyphOrder(self, glyphOrder):
        """Sets the glyph order

        Args:
                glyphOrder ([str]): List of glyph names in order.
        """
        self.glyphOrder = glyphOrder
        self._reverseGlyphOrder = {}

    def getGlyphName(self, glyphID):
        """Returns the name for the glyph with the given ID.

        Raises a ``KeyError`` if the glyph name is not found in the font.
        """
        return self.glyphOrder[glyphID]

    def _buildReverseGlyphOrderDict(self):
        self._reverseGlyphOrder = d = {}
        for glyphID, glyphName in enumerate(self.glyphOrder):
            d[glyphName] = glyphID

    def getGlyphID(self, glyphName):
        """Returns the ID of the glyph with the given name.

        Raises a ``ValueError`` if the glyph is not found in the font.
        """
        glyphOrder = self.glyphOrder
        id = getattr(self, '_reverseGlyphOrder', {}).get(glyphName)
        if id is None or id >= len(glyphOrder) or glyphOrder[id] != glyphName:
            self._buildReverseGlyphOrderDict()
            id = self._reverseGlyphOrder.get(glyphName)
        if id is None:
            raise ValueError(glyphName)
        return id

    def removeHinting(self):
        """Removes TrueType hints from all glyphs in the glyphset.

        See :py:meth:`Glyph.removeHinting`.
        """
        for glyph in self.glyphs.values():
            glyph.removeHinting()

    def keys(self):
        return self.glyphs.keys()

    def has_key(self, glyphName):
        return glyphName in self.glyphs
    __contains__ = has_key

    def get(self, glyphName, default=None):
        glyph = self.glyphs.get(glyphName, default)
        if glyph is not None:
            glyph.expand(self)
        return glyph

    def __getitem__(self, glyphName):
        glyph = self.glyphs[glyphName]
        glyph.expand(self)
        return glyph

    def __setitem__(self, glyphName, glyph):
        self.glyphs[glyphName] = glyph
        if glyphName not in self.glyphOrder:
            self.glyphOrder.append(glyphName)

    def __delitem__(self, glyphName):
        del self.glyphs[glyphName]
        self.glyphOrder.remove(glyphName)

    def __len__(self):
        assert len(self.glyphOrder) == len(self.glyphs)
        return len(self.glyphs)

    def _getPhantomPoints(self, glyphName, hMetrics, vMetrics=None):
        """Compute the four "phantom points" for the given glyph from its bounding box
        and the horizontal and vertical advance widths and sidebearings stored in the
        ttFont's "hmtx" and "vmtx" tables.

        'hMetrics' should be ttFont['hmtx'].metrics.

        'vMetrics' should be ttFont['vmtx'].metrics if there is "vmtx" or None otherwise.
        If there is no vMetrics passed in, vertical phantom points are set to the zero coordinate.

        https://docs.microsoft.com/en-us/typography/opentype/spec/tt_instructing_glyphs#phantoms
        """
        glyph = self[glyphName]
        if not hasattr(glyph, 'xMin'):
            glyph.recalcBounds(self)
        horizontalAdvanceWidth, leftSideBearing = hMetrics[glyphName]
        leftSideX = glyph.xMin - leftSideBearing
        rightSideX = leftSideX + horizontalAdvanceWidth
        if vMetrics:
            verticalAdvanceWidth, topSideBearing = vMetrics[glyphName]
            topSideY = topSideBearing + glyph.yMax
            bottomSideY = topSideY - verticalAdvanceWidth
        else:
            bottomSideY = topSideY = 0
        return [(leftSideX, 0), (rightSideX, 0), (0, topSideY), (0, bottomSideY)]

    def _getCoordinatesAndControls(self, glyphName, hMetrics, vMetrics=None, *, round=otRound):
        """Return glyph coordinates and controls as expected by "gvar" table.

        The coordinates includes four "phantom points" for the glyph metrics,
        as mandated by the "gvar" spec.

        The glyph controls is a namedtuple with the following attributes:
                - numberOfContours: -1 for composite glyphs.
                - endPts: list of indices of end points for each contour in simple
                glyphs, or component indices in composite glyphs (used for IUP
                optimization).
                - flags: array of contour point flags for simple glyphs (None for
                composite glyphs).
                - components: list of base glyph names (str) for each component in
                composite glyphs (None for simple glyphs).

        The "hMetrics" and vMetrics are used to compute the "phantom points" (see
        the "_getPhantomPoints" method).

        Return None if the requested glyphName is not present.
        """
        glyph = self.get(glyphName)
        if glyph is None:
            return None
        if glyph.isComposite():
            coords = GlyphCoordinates([(getattr(c, 'x', 0), getattr(c, 'y', 0)) for c in glyph.components])
            controls = _GlyphControls(numberOfContours=glyph.numberOfContours, endPts=list(range(len(glyph.components))), flags=None, components=[(c.glyphName, getattr(c, 'transform', None)) for c in glyph.components])
        elif glyph.isVarComposite():
            coords = []
            controls = []
            for component in glyph.components:
                componentCoords, componentControls = component.getCoordinatesAndControls()
                coords.extend(componentCoords)
                controls.extend(componentControls)
            coords = GlyphCoordinates(coords)
            controls = _GlyphControls(numberOfContours=glyph.numberOfContours, endPts=list(range(len(coords))), flags=None, components=[(c.glyphName, getattr(c, 'flags', None)) for c in glyph.components])
        else:
            coords, endPts, flags = glyph.getCoordinates(self)
            coords = coords.copy()
            controls = _GlyphControls(numberOfContours=glyph.numberOfContours, endPts=endPts, flags=flags, components=None)
        phantomPoints = self._getPhantomPoints(glyphName, hMetrics, vMetrics)
        coords.extend(phantomPoints)
        coords.toInt(round=round)
        return (coords, controls)

    def _setCoordinates(self, glyphName, coord, hMetrics, vMetrics=None):
        """Set coordinates and metrics for the given glyph.

        "coord" is an array of GlyphCoordinates which must include the "phantom
        points" as the last four coordinates.

        Both the horizontal/vertical advances and left/top sidebearings in "hmtx"
        and "vmtx" tables (if any) are updated from four phantom points and
        the glyph's bounding boxes.

        The "hMetrics" and vMetrics are used to propagate "phantom points"
        into "hmtx" and "vmtx" tables if desired.  (see the "_getPhantomPoints"
        method).
        """
        glyph = self[glyphName]
        assert len(coord) >= 4
        leftSideX = coord[-4][0]
        rightSideX = coord[-3][0]
        topSideY = coord[-2][1]
        bottomSideY = coord[-1][1]
        coord = coord[:-4]
        if glyph.isComposite():
            assert len(coord) == len(glyph.components)
            for p, comp in zip(coord, glyph.components):
                if hasattr(comp, 'x'):
                    comp.x, comp.y = p
        elif glyph.isVarComposite():
            for comp in glyph.components:
                coord = comp.setCoordinates(coord)
            assert not coord
        elif glyph.numberOfContours == 0:
            assert len(coord) == 0
        else:
            assert len(coord) == len(glyph.coordinates)
            glyph.coordinates = GlyphCoordinates(coord)
        glyph.recalcBounds(self, boundsDone=set())
        horizontalAdvanceWidth = otRound(rightSideX - leftSideX)
        if horizontalAdvanceWidth < 0:
            horizontalAdvanceWidth = 0
        leftSideBearing = otRound(glyph.xMin - leftSideX)
        hMetrics[glyphName] = (horizontalAdvanceWidth, leftSideBearing)
        if vMetrics is not None:
            verticalAdvanceWidth = otRound(topSideY - bottomSideY)
            if verticalAdvanceWidth < 0:
                verticalAdvanceWidth = 0
            topSideBearing = otRound(topSideY - glyph.yMax)
            vMetrics[glyphName] = (verticalAdvanceWidth, topSideBearing)

    def _synthesizeVMetrics(self, glyphName, ttFont, defaultVerticalOrigin):
        """This method is wrong and deprecated.
        For rationale see:
        https://github.com/fonttools/fonttools/pull/2266/files#r613569473
        """
        vMetrics = getattr(ttFont.get('vmtx'), 'metrics', None)
        if vMetrics is None:
            verticalAdvanceWidth = ttFont['head'].unitsPerEm
            topSideY = getattr(ttFont.get('hhea'), 'ascent', None)
            if topSideY is None:
                if defaultVerticalOrigin is not None:
                    topSideY = defaultVerticalOrigin
                else:
                    topSideY = verticalAdvanceWidth
            glyph = self[glyphName]
            glyph.recalcBounds(self)
            topSideBearing = otRound(topSideY - glyph.yMax)
            vMetrics = {glyphName: (verticalAdvanceWidth, topSideBearing)}
        return vMetrics

    @deprecateFunction("use '_getPhantomPoints' instead", category=DeprecationWarning)
    def getPhantomPoints(self, glyphName, ttFont, defaultVerticalOrigin=None):
        """Old public name for self._getPhantomPoints().
        See: https://github.com/fonttools/fonttools/pull/2266"""
        hMetrics = ttFont['hmtx'].metrics
        vMetrics = self._synthesizeVMetrics(glyphName, ttFont, defaultVerticalOrigin)
        return self._getPhantomPoints(glyphName, hMetrics, vMetrics)

    @deprecateFunction("use '_getCoordinatesAndControls' instead", category=DeprecationWarning)
    def getCoordinatesAndControls(self, glyphName, ttFont, defaultVerticalOrigin=None):
        """Old public name for self._getCoordinatesAndControls().
        See: https://github.com/fonttools/fonttools/pull/2266"""
        hMetrics = ttFont['hmtx'].metrics
        vMetrics = self._synthesizeVMetrics(glyphName, ttFont, defaultVerticalOrigin)
        return self._getCoordinatesAndControls(glyphName, hMetrics, vMetrics)

    @deprecateFunction("use '_setCoordinates' instead", category=DeprecationWarning)
    def setCoordinates(self, glyphName, ttFont):
        """Old public name for self._setCoordinates().
        See: https://github.com/fonttools/fonttools/pull/2266"""
        hMetrics = ttFont['hmtx'].metrics
        vMetrics = getattr(ttFont.get('vmtx'), 'metrics', None)
        self._setCoordinates(glyphName, hMetrics, vMetrics)