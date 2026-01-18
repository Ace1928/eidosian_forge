import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
@classmethod
def _find_data_repr_(mcls, class_name, bases):
    for chain in bases:
        for base in chain.__mro__:
            if base is object:
                continue
            elif isinstance(base, EnumType):
                return base._value_repr_
            elif '__repr__' in base.__dict__:
                return base.__dict__['__repr__']
    return None