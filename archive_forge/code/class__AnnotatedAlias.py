import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
class _AnnotatedAlias(typing._GenericAlias, _root=True):
    """Runtime representation of an annotated type.

        At its core 'Annotated[t, dec1, dec2, ...]' is an alias for the type 't'
        with extra annotations. The alias behaves like a normal typing alias,
        instantiating is the same as instantiating the underlying type, binding
        it to types is also the same.
        """

    def __init__(self, origin, metadata):
        if isinstance(origin, _AnnotatedAlias):
            metadata = origin.__metadata__ + metadata
            origin = origin.__origin__
        super().__init__(origin, origin)
        self.__metadata__ = metadata

    def copy_with(self, params):
        assert len(params) == 1
        new_type = params[0]
        return _AnnotatedAlias(new_type, self.__metadata__)

    def __repr__(self):
        return f'typing_extensions.Annotated[{typing._type_repr(self.__origin__)}, {', '.join((repr(a) for a in self.__metadata__))}]'

    def __reduce__(self):
        return (operator.getitem, (Annotated, (self.__origin__,) + self.__metadata__))

    def __eq__(self, other):
        if not isinstance(other, _AnnotatedAlias):
            return NotImplemented
        if self.__origin__ != other.__origin__:
            return False
        return self.__metadata__ == other.__metadata__

    def __hash__(self):
        return hash((self.__origin__, self.__metadata__))