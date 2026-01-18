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
def addCFFVarStore(varFont, varModel, varDataList, masterSupports):
    fvarTable = varFont['fvar']
    axisKeys = [axis.axisTag for axis in fvarTable.axes]
    varTupleList = varLib.builder.buildVarRegionList(masterSupports, axisKeys)
    varStoreCFFV = varLib.builder.buildVarStore(varTupleList, varDataList)
    topDict = varFont['CFF2'].cff.topDictIndex[0]
    topDict.VarStore = VarStoreData(otVarStore=varStoreCFFV)
    if topDict.FDArray[0].vstore is None:
        fdArray = topDict.FDArray
        for fontDict in fdArray:
            if hasattr(fontDict, 'Private'):
                fontDict.Private.vstore = topDict.VarStore