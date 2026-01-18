import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
class _auto_null:

    def __repr__(self):
        return '_auto_null'