import collections
import enum
from fontTools.ttLib.tables.otBase import (
from fontTools.ttLib.tables.otConverters import (
from fontTools.misc.roundTools import otRound
def _assignable(convertersByName):
    return {k: v for k, v in convertersByName.items() if not isinstance(v, ComputedInt)}