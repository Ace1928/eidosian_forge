import abc
import collections
import collections.abc
import operator
import sys
import typing
class _ConcatenateAliasBase(typing._FinalTypingBase, metaclass=_ConcatenateAliasMeta, _root=True):
    """Used in conjunction with ``ParamSpec`` and ``Callable`` to represent a
        higher order function which adds, removes or transforms parameters of a
        callable.

        For example::

           Callable[Concatenate[int, P], int]

        See PEP 612 for detailed information.
        """
    __slots__ = ()

    def __instancecheck__(self, obj):
        raise TypeError('Concatenate cannot be used with isinstance().')

    def __subclasscheck__(self, cls):
        raise TypeError('Concatenate cannot be used with issubclass().')

    def __repr__(self):
        return 'typing_extensions.Concatenate'

    def __getitem__(self, parameters):
        return _concatenate_getitem(self, parameters)