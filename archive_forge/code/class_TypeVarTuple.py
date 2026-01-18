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
class TypeVarTuple(_DefaultMixin):
    """Type variable tuple.

        Usage::

            Ts = TypeVarTuple('Ts')

        In the same way that a normal type variable is a stand-in for a single
        type such as ``int``, a type variable *tuple* is a stand-in for a *tuple*
        type such as ``Tuple[int, str]``.

        Type variable tuples can be used in ``Generic`` declarations.
        Consider the following example::

            class Array(Generic[*Ts]): ...

        The ``Ts`` type variable tuple here behaves like ``tuple[T1, T2]``,
        where ``T1`` and ``T2`` are type variables. To use these type variables
        as type parameters of ``Array``, we must *unpack* the type variable tuple using
        the star operator: ``*Ts``. The signature of ``Array`` then behaves
        as if we had simply written ``class Array(Generic[T1, T2]): ...``.
        In contrast to ``Generic[T1, T2]``, however, ``Generic[*Shape]`` allows
        us to parameterise the class with an *arbitrary* number of type parameters.

        Type variable tuples can be used anywhere a normal ``TypeVar`` can.
        This includes class definitions, as shown above, as well as function
        signatures and variable annotations::

            class Array(Generic[*Ts]):

                def __init__(self, shape: Tuple[*Ts]):
                    self._shape: Tuple[*Ts] = shape

                def get_shape(self) -> Tuple[*Ts]:
                    return self._shape

            shape = (Height(480), Width(640))
            x: Array[Height, Width] = Array(shape)
            y = abs(x)  # Inferred type is Array[Height, Width]
            z = x + x   #        ...    is Array[Height, Width]
            x.get_shape()  #     ...    is tuple[Height, Width]

        """
    __class__ = typing.TypeVar

    def __iter__(self):
        yield self.__unpacked__

    def __init__(self, name, *, default=_marker):
        self.__name__ = name
        _DefaultMixin.__init__(self, default)
        def_mod = _caller()
        if def_mod != 'typing_extensions':
            self.__module__ = def_mod
        self.__unpacked__ = Unpack[self]

    def __repr__(self):
        return self.__name__

    def __hash__(self):
        return object.__hash__(self)

    def __eq__(self, other):
        return self is other

    def __reduce__(self):
        return self.__name__

    def __init_subclass__(self, *args, **kwds):
        if '_root' not in kwds:
            raise TypeError('Cannot subclass special typing classes')