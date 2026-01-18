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
def _xmlWriteLigatures(self, xmlWriter, font, value, name, attrs):
    if not hasattr(value, 'Ligatures'):
        return
    xmlWriter.begintag('Ligatures')
    xmlWriter.newline()
    for i, g in enumerate(getattr(value, 'Ligatures')):
        xmlWriter.simpletag('Ligature', index=i, glyph=g)
        xmlWriter.newline()
    xmlWriter.endtag('Ligatures')
    xmlWriter.newline()