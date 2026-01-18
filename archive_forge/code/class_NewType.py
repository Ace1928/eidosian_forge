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
class NewType:
    """NewType creates simple unique types with almost zero
        runtime overhead. NewType(name, tp) is considered a subtype of tp
        by static type checkers. At runtime, NewType(name, tp) returns
        a dummy callable that simply returns its argument. Usage::
            UserId = NewType('UserId', int)
            def name_by_id(user_id: UserId) -> str:
                ...
            UserId('user')          # Fails type check
            name_by_id(42)          # Fails type check
            name_by_id(UserId(42))  # OK
            num = UserId(5) + 1     # type: int
        """

    def __call__(self, obj, /):
        return obj

    def __init__(self, name, tp):
        self.__qualname__ = name
        if '.' in name:
            name = name.rpartition('.')[-1]
        self.__name__ = name
        self.__supertype__ = tp
        def_mod = _caller()
        if def_mod != 'typing_extensions':
            self.__module__ = def_mod

    def __mro_entries__(self, bases):
        supercls_name = self.__name__

        class Dummy:

            def __init_subclass__(cls):
                subcls_name = cls.__name__
                raise TypeError(f'Cannot subclass an instance of NewType. Perhaps you were looking for: `{subcls_name} = NewType({subcls_name!r}, {supercls_name})`')
        return (Dummy,)

    def __repr__(self):
        return f'{self.__module__}.{self.__qualname__}'

    def __reduce__(self):
        return self.__qualname__
    if sys.version_info >= (3, 10):

        def __or__(self, other):
            return typing.Union[self, other]

        def __ror__(self, other):
            return typing.Union[other, self]