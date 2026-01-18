import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
@_simple_enum(StrEnum)
class EnumCheck:
    """
    various conditions to check an enumeration for
    """
    CONTINUOUS = 'no skipped integer values'
    NAMED_FLAGS = 'multi-flag aliases may not contain unnamed flags'
    UNIQUE = 'one name per value'