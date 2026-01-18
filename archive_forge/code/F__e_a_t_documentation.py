from fontTools.misc import sstruct
from fontTools.misc.fixedTools import floatToFixedToStr
from fontTools.misc.textTools import safeEval
from . import DefaultTable
from . import grUtils
import struct
The ``Feat`` table is used exclusively by the Graphite shaping engine
    to store features and possible settings specified in GDL. Graphite features
    determine what rules are applied to transform a glyph stream.

    Not to be confused with ``feat``, or the OpenType Layout tables
    ``GSUB``/``GPOS``.