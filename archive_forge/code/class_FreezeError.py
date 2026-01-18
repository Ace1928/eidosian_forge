from types import MappingProxyType
from array import array
from frozendict import frozendict
import warnings
from collections.abc import MutableMapping, MutableSequence, MutableSet
class FreezeError(Exception):
    pass