from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import nearestMultipleShortestRepr, otRound
from fontTools.misc.textTools import bytesjoin, tobytes, tostr, pad, safeEval
from fontTools.ttLib import getSearchRange
from .otBase import (
from .otTables import (
from itertools import zip_longest
from functools import partial
import re
import struct
from typing import Optional
import logging
def _compileStates(self, font, states, glyphClassCount, actionIndex):
    stateArrayWriter = OTTableWriter()
    entries, entryIDs = ([], {})
    for state in states:
        for glyphClass in range(glyphClassCount):
            transition = state.Transitions[glyphClass]
            entryWriter = OTTableWriter()
            transition.compile(entryWriter, font, actionIndex)
            entryData = entryWriter.getAllData()
            assert len(entryData) == transition.staticSize, '%s has staticSize %d, but actually wrote %d bytes' % (repr(transition), transition.staticSize, len(entryData))
            entryIndex = entryIDs.get(entryData)
            if entryIndex is None:
                entryIndex = len(entries)
                entryIDs[entryData] = entryIndex
                entries.append(entryData)
            stateArrayWriter.writeUShort(entryIndex)
    stateArrayData = pad(stateArrayWriter.getAllData(), 4)
    entryTableData = pad(bytesjoin(entries), 4)
    return (stateArrayData, entryTableData)