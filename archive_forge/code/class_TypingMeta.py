import abc
from abc import abstractmethod, abstractproperty
import collections
import contextlib
import functools
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
class TypingMeta(type):
    """Metaclass for most types defined in typing module
    (not a part of public API).

    This overrides __new__() to require an extra keyword parameter
    '_root', which serves as a guard against naive subclassing of the
    typing classes.  Any legitimate class defined using a metaclass
    derived from TypingMeta must pass _root=True.

    This also defines a dummy constructor (all the work for most typing
    constructs is done in __new__) and a nicer repr().
    """
    _is_protocol = False

    def __new__(cls, name, bases, namespace, *, _root=False):
        if not _root:
            raise TypeError('Cannot subclass %s' % (', '.join(map(_type_repr, bases)) or '()'))
        return super().__new__(cls, name, bases, namespace)

    def __init__(self, *args, **kwds):
        pass

    def _eval_type(self, globalns, localns):
        """Override this in subclasses to interpret forward references.

        For example, List['C'] is internally stored as
        List[_ForwardRef('C')], which should evaluate to List[C],
        where C is an object found in globalns or localns (searching
        localns first, of course).
        """
        return self

    def _get_type_vars(self, tvars):
        pass

    def __repr__(self):
        qname = _trim_name(_qualname(self))
        return '%s.%s' % (self.__module__, qname)