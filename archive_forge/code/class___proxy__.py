import copy
import itertools
import operator
from functools import wraps
class __proxy__(Promise):
    """
        Encapsulate a function call and act as a proxy for methods that are
        called on the result of that function. The function is not evaluated
        until one of the methods on the result is called.
        """

    def __init__(self, args, kw):
        self._args = args
        self._kw = kw

    def __reduce__(self):
        return (_lazy_proxy_unpickle, (func, self._args, self._kw) + resultclasses)

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def __cast(self):
        return func(*self._args, **self._kw)

    def __repr__(self):
        return repr(self.__cast())

    def __str__(self):
        return str(self.__cast())

    def __eq__(self, other):
        if isinstance(other, Promise):
            other = other.__cast()
        return self.__cast() == other

    def __ne__(self, other):
        if isinstance(other, Promise):
            other = other.__cast()
        return self.__cast() != other

    def __lt__(self, other):
        if isinstance(other, Promise):
            other = other.__cast()
        return self.__cast() < other

    def __le__(self, other):
        if isinstance(other, Promise):
            other = other.__cast()
        return self.__cast() <= other

    def __gt__(self, other):
        if isinstance(other, Promise):
            other = other.__cast()
        return self.__cast() > other

    def __ge__(self, other):
        if isinstance(other, Promise):
            other = other.__cast()
        return self.__cast() >= other

    def __hash__(self):
        return hash(self.__cast())

    def __format__(self, format_spec):
        return format(self.__cast(), format_spec)

    def __add__(self, other):
        return self.__cast() + other

    def __radd__(self, other):
        return other + self.__cast()

    def __mod__(self, other):
        return self.__cast() % other

    def __mul__(self, other):
        return self.__cast() * other