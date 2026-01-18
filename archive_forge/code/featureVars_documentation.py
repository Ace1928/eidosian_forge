from fontTools.misc.dictTools import hashdict
from fontTools.misc.intTools import bit_count
from fontTools.ttLib import newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.ttVisitor import TTVisitor
from fontTools.otlLib.builder import buildLookup, buildSingleSubstSubtable
from collections import OrderedDict
from .errors import VarLibError, VarLibValidationError
Go through the scripts list, and remap feature indices.