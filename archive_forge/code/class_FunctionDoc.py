import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
class FunctionDoc(NumpyDocString):

    def __init__(self, func, role='func', doc=None, config={}):
        self._f = func
        self._role = role
        if doc is None:
            if func is None:
                raise ValueError('No function or docstring given')
            doc = inspect.getdoc(func) or ''
        NumpyDocString.__init__(self, doc, config)

    def get_func(self):
        func_name = getattr(self._f, '__name__', self.__class__.__name__)
        if inspect.isclass(self._f):
            func = getattr(self._f, '__call__', self._f.__init__)
        else:
            func = self._f
        return (func, func_name)

    def __str__(self):
        out = ''
        func, func_name = self.get_func()
        roles = {'func': 'function', 'meth': 'method'}
        if self._role:
            if self._role not in roles:
                print('Warning: invalid role %s' % self._role)
            out += '.. {}:: {}\n    \n\n'.format(roles.get(self._role, ''), func_name)
        out += super().__str__(func_role=self._role)
        return out