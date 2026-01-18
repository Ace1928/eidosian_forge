import locale
from gettext import NullTranslations, translation
from os import path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
class _TranslationProxy:
    """
    Class for proxy strings from gettext translations. This is a helper for the
    lazy_* functions from this module.

    The proxy implementation attempts to be as complete as possible, so that
    the lazy objects should mostly work as expected, for example for sorting.
    """
    __slots__ = ('_func', '_args')

    def __new__(cls, func: Callable[..., str], *args: str) -> '_TranslationProxy':
        if not args:
            return str(func)
        return object.__new__(cls)

    def __getnewargs__(self) -> Tuple[str]:
        return (self._func,) + self._args

    def __init__(self, func: Callable[..., str], *args: str) -> None:
        self._func = func
        self._args = args

    def __str__(self) -> str:
        return str(self._func(*self._args))

    def __dir__(self) -> List[str]:
        return dir(str)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__str__(), name)

    def __getstate__(self) -> Tuple[Callable[..., str], Tuple[str, ...]]:
        return (self._func, self._args)

    def __setstate__(self, tup: Tuple[Callable[..., str], Tuple[str]]) -> None:
        self._func, self._args = tup

    def __copy__(self) -> '_TranslationProxy':
        return _TranslationProxy(self._func, *self._args)

    def __repr__(self) -> str:
        try:
            return 'i' + repr(str(self.__str__()))
        except Exception:
            return f'<{self.__class__.__name__} broken>'

    def __add__(self, other: str) -> str:
        return self.__str__() + other

    def __radd__(self, other: str) -> str:
        return other + self.__str__()

    def __mod__(self, other: str) -> str:
        return self.__str__() % other

    def __rmod__(self, other: str) -> str:
        return other % self.__str__()

    def __mul__(self, other: Any) -> str:
        return self.__str__() * other

    def __rmul__(self, other: Any) -> str:
        return other * self.__str__()

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        return self.__str__() == other

    def __lt__(self, string):
        return self.__str__() < string

    def __contains__(self, char):
        return char in self.__str__()

    def __len__(self):
        return len(self.__str__())

    def __getitem__(self, index):
        return self.__str__()[index]