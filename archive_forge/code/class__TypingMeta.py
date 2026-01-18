import abc
import sys
import types
from collections.abc import Mapping, MutableMapping
class _TypingMeta(abc.ABCMeta):
    if sys.version_info >= (3, 9):

        def __getitem__(self, key):
            return types.GenericAlias(self, key)
    else:

        def __getitem__(self, key):
            return self