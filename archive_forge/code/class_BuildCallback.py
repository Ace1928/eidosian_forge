import collections
import enum
from fontTools.ttLib.tables.otBase import (
from fontTools.ttLib.tables.otConverters import (
from fontTools.misc.roundTools import otRound
class BuildCallback(enum.Enum):
    """Keyed on (BEFORE_BUILD, class[, Format if available]).
    Receives (dest, source).
    Should return (dest, source), which can be new objects.
    """
    BEFORE_BUILD = enum.auto()
    'Keyed on (AFTER_BUILD, class[, Format if available]).\n    Receives (dest).\n    Should return dest, which can be a new object.\n    '
    AFTER_BUILD = enum.auto()
    'Keyed on (CREATE_DEFAULT, class[, Format if available]).\n    Receives no arguments.\n    Should return a new instance of class.\n    '
    CREATE_DEFAULT = enum.auto()