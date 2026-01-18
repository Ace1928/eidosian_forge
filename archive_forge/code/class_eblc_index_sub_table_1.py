from fontTools.misc import sstruct
from . import DefaultTable
from fontTools.misc.textTools import bytesjoin, safeEval
from .BitmapGlyphMetrics import (
import struct
import itertools
from collections import deque
import logging
class eblc_index_sub_table_1(_createOffsetArrayIndexSubTableMixin('L'), EblcIndexSubTable):
    pass