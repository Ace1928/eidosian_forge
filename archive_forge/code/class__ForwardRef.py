import abc
from abc import abstractmethod, abstractproperty
import collections
import contextlib
import functools
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
class _ForwardRef(_TypingBase, _root=True):
    """Internal wrapper to hold a forward reference."""
    __slots__ = ('__forward_arg__', '__forward_code__', '__forward_evaluated__', '__forward_value__')

    def __init__(self, arg):
        super().__init__(arg)
        if not isinstance(arg, str):
            raise TypeError('Forward reference must be a string -- got %r' % (arg,))
        try:
            code = compile(arg, '<string>', 'eval')
        except SyntaxError:
            raise SyntaxError('Forward reference must be an expression -- got %r' % (arg,))
        self.__forward_arg__ = arg
        self.__forward_code__ = code
        self.__forward_evaluated__ = False
        self.__forward_value__ = None

    def _eval_type(self, globalns, localns):
        if not self.__forward_evaluated__ or localns is not globalns:
            if globalns is None and localns is None:
                globalns = localns = {}
            elif globalns is None:
                globalns = localns
            elif localns is None:
                localns = globalns
            self.__forward_value__ = _type_check(eval(self.__forward_code__, globalns, localns), 'Forward references must evaluate to types.')
            self.__forward_evaluated__ = True
        return self.__forward_value__

    def __eq__(self, other):
        if not isinstance(other, _ForwardRef):
            return NotImplemented
        if self.__forward_evaluated__ and other.__forward_evaluated__:
            return self.__forward_arg__ == other.__forward_arg__ and self.__forward_value__ == other.__forward_value__
        return self.__forward_arg__ == other.__forward_arg__

    def __hash__(self):
        return hash(self.__forward_arg__)

    def __instancecheck__(self, obj):
        raise TypeError('Forward references cannot be used with isinstance().')

    def __subclasscheck__(self, cls):
        raise TypeError('Forward references cannot be used with issubclass().')

    def __repr__(self):
        return '_ForwardRef(%r)' % (self.__forward_arg__,)