from collections import UserDict, deque
from functools import partial
from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
from . import DefaultTable
import array
import itertools
import logging
import struct
import sys
import fontTools.ttLib.tables.TupleVariation as tv
Packs a list of offsets into a 'gvar' offset table.

        Returns a pair (bytestring, tableFormat). Bytestring is the
        packed offset table. Format indicates whether the table
        uses short (tableFormat=0) or long (tableFormat=1) integers.
        The returned tableFormat should get packed into the flags field
        of the 'gvar' header.
        