from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
class _Call(tuple):
    """
    A tuple for holding the results of a call to a mock, either in the form
    `(args, kwargs)` or `(name, args, kwargs)`.

    If args or kwargs are empty then a call tuple will compare equal to
    a tuple without those values. This makes comparisons less verbose::

        _Call(('name', (), {})) == ('name',)
        _Call(('name', (1,), {})) == ('name', (1,))
        _Call(((), {'a': 'b'})) == ({'a': 'b'},)

    The `_Call` object provides a useful shortcut for comparing with call::

        _Call(((1, 2), {'a': 3})) == call(1, 2, a=3)
        _Call(('foo', (1, 2), {'a': 3})) == call.foo(1, 2, a=3)

    If the _Call has no name then it will match any name.
    """

    def __new__(cls, value=(), name=None, parent=None, two=False, from_kall=True):
        name = ''
        args = ()
        kwargs = {}
        _len = len(value)
        if _len == 3:
            name, args, kwargs = value
        elif _len == 2:
            first, second = value
            if isinstance(first, basestring):
                name = first
                if isinstance(second, tuple):
                    args = second
                else:
                    kwargs = second
            else:
                args, kwargs = (first, second)
        elif _len == 1:
            value, = value
            if isinstance(value, basestring):
                name = value
            elif isinstance(value, tuple):
                args = value
            else:
                kwargs = value
        if two:
            return tuple.__new__(cls, (args, kwargs))
        return tuple.__new__(cls, (name, args, kwargs))

    def __init__(self, value=(), name=None, parent=None, two=False, from_kall=True):
        self.name = name
        self.parent = parent
        self.from_kall = from_kall

    def __eq__(self, other):
        if other is ANY:
            return True
        try:
            len_other = len(other)
        except TypeError:
            return False
        self_name = ''
        if len(self) == 2:
            self_args, self_kwargs = self
        else:
            self_name, self_args, self_kwargs = self
        other_name = ''
        if len_other == 0:
            other_args, other_kwargs = ((), {})
        elif len_other == 3:
            other_name, other_args, other_kwargs = other
        elif len_other == 1:
            value, = other
            if isinstance(value, tuple):
                other_args = value
                other_kwargs = {}
            elif isinstance(value, basestring):
                other_name = value
                other_args, other_kwargs = ((), {})
            else:
                other_args = ()
                other_kwargs = value
        elif len_other == 2:
            first, second = other
            if isinstance(first, basestring):
                other_name = first
                if isinstance(second, tuple):
                    other_args, other_kwargs = (second, {})
                else:
                    other_args, other_kwargs = ((), second)
            else:
                other_args, other_kwargs = (first, second)
        else:
            return False
        if self_name and other_name != self_name:
            return False
        return (other_args, other_kwargs) == (self_args, self_kwargs)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __call__(self, *args, **kwargs):
        if self.name is None:
            return _Call(('', args, kwargs), name='()')
        name = self.name + '()'
        return _Call((self.name, args, kwargs), name=name, parent=self)

    def __getattr__(self, attr):
        if self.name is None:
            return _Call(name=attr, from_kall=False)
        name = '%s.%s' % (self.name, attr)
        return _Call(name=name, parent=self, from_kall=False)

    def count(self, *args, **kwargs):
        return self.__getattr__('count')(*args, **kwargs)

    def index(self, *args, **kwargs):
        return self.__getattr__('index')(*args, **kwargs)

    def __repr__(self):
        if not self.from_kall:
            name = self.name or 'call'
            if name.startswith('()'):
                name = 'call%s' % name
            return name
        if len(self) == 2:
            name = 'call'
            args, kwargs = self
        else:
            name, args, kwargs = self
            if not name:
                name = 'call'
            elif not name.startswith('()'):
                name = 'call.%s' % name
            else:
                name = 'call%s' % name
        return _format_call_signature(name, args, kwargs)

    def call_list(self):
        """For a call object that represents multiple calls, `call_list`
        returns a list of all the intermediate calls as well as the
        final call."""
        vals = []
        thing = self
        while thing is not None:
            if thing.from_kall:
                vals.append(thing)
            thing = thing.parent
        return _CallList(reversed(vals))