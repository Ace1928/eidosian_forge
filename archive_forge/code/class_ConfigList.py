import argparse
import builtins
import enum
import importlib
import inspect
import io
import logging
import os
import pickle
import ply.lex
import re
import sys
import textwrap
import types
from operator import attrgetter
from pyomo.common.collections import Sequence, Mapping
from pyomo.common.deprecation import (
from pyomo.common.fileutils import import_file
from pyomo.common.formatting import wrap_reStructuredText
from pyomo.common.modeling import NOTSET
class ConfigList(ConfigBase, Sequence):
    """Store and manipulate a list of configuration values.

    Parameters
    ----------
    default: optional
        The default value that this ConfigList will take if no value is
        provided.  If default is a list or ConfigList, then each member
        is cast to the ConfigList's domain to build the default value,
        otherwise the default is cast to the domain and forms a default
        list with a single element.

    domain: Callable, optional
        The domain can be any callable that accepts a candidate value
        and returns the value converted to the desired type, optionally
        performing any data validation.  The result will be stored /
        added to the ConfigList.  Examples include type constructors
        like `int` or `float`.  More complex domain examples include
        callable objects; for example, the :py:class:`In` class that
        ensures that the value falls into an acceptable set or even a
        complete :py:class:`ConfigDict` instance.

    description: str, optional
        The short description of this list

    doc: str, optional
        The long documentation string for this list

    visibility: int, optional
        The visibility of this ConfigList when generating templates and
        documentation.  Visibility supports specification of "advanced"
        or "developer" options.  ConfigLists with visibility=0 (the
        default) will always be printed / included.  ConfigLists
        with higher visibility values will only be included when the
        generation method specifies a visibility greater than or equal
        to the visibility of this object.

    """

    def __init__(self, *args, **kwds):
        ConfigBase.__init__(self, *args, **kwds)
        if self._domain is None:
            self._domain = ConfigValue()
        elif isinstance(self._domain, ConfigBase):
            pass
        else:
            self._domain = ConfigValue(None, domain=self._domain)
        self.reset()

    def __setstate__(self, state):
        state = super(ConfigList, self).__setstate__(state)
        for x in self._data:
            x._parent = self

    def __getitem__(self, key):
        val = self._data[key]
        self._userAccessed = True
        if isinstance(val, ConfigValue):
            return val.value()
        else:
            return val

    def get(self, key, default=NOTSET):
        try:
            val = self._data[key]
            self._userAccessed = True
            return val
        except IndexError:
            if default is NOTSET:
                raise
        return self._domain(default)

    def __setitem__(self, key, val):
        self._data[key].set_value(val)

    def __len__(self):
        return self._data.__len__()

    def __iter__(self):
        self._userAccessed = True
        return iter((self[i] for i in range(len(self._data))))

    def value(self, accessValue=True):
        if accessValue:
            self._userAccessed = True
        return [config.value(accessValue) for config in self._data]

    def set_value(self, value):
        _old = self._data
        self._data = []
        try:
            if isinstance(value, str):
                value = list(_default_string_list_lexer(value))
            if type(value) is list or isinstance(value, ConfigList):
                for val in value:
                    self.append(val)
            else:
                self.append(value)
        except:
            self._data = _old
            raise
        self._userSet = True

    def reset(self):
        ConfigBase.reset(self)
        for val in self.user_values():
            val._userSet = False

    def append(self, value=NOTSET):
        val = self._cast(value)
        if val is None:
            return
        self._data.append(val)
        self._data[-1]._parent = self
        self._data[-1]._name = '[%s]' % (len(self._data) - 1,)
        self._data[-1]._userSet = True

    @deprecated('ConfigList.add() has been deprecated.  Use append()', version='5.7.2')
    def add(self, value=NOTSET):
        """Append the specified value to the list, casting as necessary."""
        return self.append(value)

    def _data_collector(self, level, prefix, visibility=None, docMode=False):
        if visibility is not None and visibility < self._visibility:
            return
        if docMode:
            yield (level, prefix, None, self)
            subDomain = self._domain._data_collector(level + 1, '- ', visibility, docMode)
            next(subDomain)
            for v in subDomain:
                yield v
            return
        if prefix:
            if not self._data:
                yield (level, prefix, [], self)
            else:
                yield (level, prefix, None, self)
                if level is not None:
                    level += 1
        for value in self._data:
            for v in value._data_collector(level, '- ', visibility, docMode):
                yield v