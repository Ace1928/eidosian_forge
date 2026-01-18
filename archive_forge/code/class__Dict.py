from fontTools.misc import sstruct
from fontTools.misc.fixedTools import floatToFixedToStr
from fontTools.misc.textTools import safeEval
from functools import partial
from . import DefaultTable
from . import grUtils
import struct
class _Dict(dict):
    pass