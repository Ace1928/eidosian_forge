import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
class IntEnum(int, ReprEnum):
    """
    Enum where members are also (and must be) ints
    """