from __future__ import annotations
from fontTools.misc.textTools import num2binary, binary2num, readHex, strjoin
import array
from io import StringIO
from typing import List
import re
import logging
def _skipWhite(data, pos):
    m = _whiteRE.match(data, pos)
    newPos = m.regs[0][1]
    assert newPos >= pos
    return newPos