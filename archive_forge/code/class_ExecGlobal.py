from __future__ import unicode_literals
import collections
import contextlib
import inspect
import logging
import pprint
import sys
import textwrap
import six
class ExecGlobal(dict):
    """
  We pass this in as the "global" variable when parsing a config. It is
  composed of a nested dictionary and provides methods for mangaging
  the current  "stack" into that nested dictionary
  """

    def __init__(self, filepath):
        super(ExecGlobal, self).__init__()
        self._dict_stack = []
        self.filepath = filepath

    @contextlib.contextmanager
    def section(self, key):
        self.push_section(key)
        yield None
        self.pop_section()

    def push_section(self, key):
        key = key.replace('-', '_')
        if self._dict_stack:
            stacktop = self._dict_stack[-1]
        else:
            stacktop = self
        if key not in stacktop:
            stacktop[key] = {}
        newtop = stacktop[key]
        self._dict_stack.append(newtop)

    def pop_section(self):
        self._dict_stack.pop(-1)

    def __getitem__(self, key):
        if key in ('_dict_stack', '__name__'):
            logger.warning('Config illegal attempt to access %s', key)
            return None
        if key == '__file__':
            return self.filepath
        if hasattr(self, key):
            return getattr(self, key)
        for dictitem in reversed(self._dict_stack):
            if key in dictitem:
                return dictitem[key]
        return super(ExecGlobal, self).__getitem__(key)

    def __setitem__(self, key, value):
        if self._dict_stack:
            self._dict_stack[-1][key] = value
        else:
            super(ExecGlobal, self).__setitem__(key, value)