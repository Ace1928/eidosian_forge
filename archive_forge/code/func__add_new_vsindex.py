from collections import namedtuple
from fontTools.cffLib import (
from io import BytesIO
from fontTools.cffLib.specializer import specializeCommands, commandsToProgram
from fontTools.ttLib import newTable
from fontTools import varLib
from fontTools.varLib.models import allEqual
from fontTools.misc.roundTools import roundFunc
from fontTools.misc.psCharStrings import T2CharString, T2OutlineExtractor
from fontTools.pens.t2CharStringPen import T2CharStringPen
from functools import partial
from .errors import (
def _add_new_vsindex(model, key, masterSupports, vsindex_dict, vsindex_by_key, varDataList):
    varTupleIndexes = []
    for support in model.supports[1:]:
        if support not in masterSupports:
            masterSupports.append(support)
        varTupleIndexes.append(masterSupports.index(support))
    var_data = varLib.builder.buildVarData(varTupleIndexes, None, False)
    vsindex = len(vsindex_dict)
    vsindex_by_key[key] = vsindex
    vsindex_dict[vsindex] = (model, [key])
    varDataList.append(var_data)
    return vsindex