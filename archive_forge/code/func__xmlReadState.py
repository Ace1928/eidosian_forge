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
def _xmlReadState(self, attrs, content, font):
    state = AATState()
    for eltName, eltAttrs, eltContent in filter(istuple, content):
        if eltName == 'Transition':
            glyphClass = safeEval(eltAttrs['onGlyphClass'])
            transition = self.tableClass()
            transition.fromXML(eltName, eltAttrs, eltContent, font)
            state.Transitions[glyphClass] = transition
    return state