import abc
import collections
import collections.abc
import operator
import sys
import typing
class _ConcatenateAliasMeta(typing.TypingMeta):
    """Metaclass for Concatenate."""

    def __repr__(self):
        return 'typing_extensions.Concatenate'