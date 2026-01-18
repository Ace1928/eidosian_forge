import abc
import collections
import collections.abc
import operator
import sys
import typing
class _TypeAliasMeta(typing.TypingMeta):
    """Metaclass for TypeAlias"""

    def __repr__(self):
        return 'typing_extensions.TypeAlias'