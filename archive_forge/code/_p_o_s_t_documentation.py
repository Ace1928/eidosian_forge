from fontTools import ttLib
from fontTools.ttLib.standardGlyphOrder import standardGlyphOrder
from fontTools.misc import sstruct
from fontTools.misc.textTools import bytechr, byteord, tobytes, tostr, safeEval, readHex
from . import DefaultTable
import sys
import struct
import array
import logging
This function will get called by a ttLib.TTFont instance.
        Do not call this function yourself, use TTFont().getGlyphOrder()
        or its relatives instead!
        