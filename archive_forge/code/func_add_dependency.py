import contextlib
import errno
import re
import sys
import typing
from abc import ABC
from collections import OrderedDict
from collections.abc import MutableMapping
from types import TracebackType
from typing import Dict, Set, Optional, Union, Iterator, IO, Iterable, TYPE_CHECKING, Type
def add_dependency(self, substvar, dependency_clause):
    """Add a dependency clause to a given substvar

        >>> substvars = Substvars()
        >>> # add_dependency automatically creates variables
        >>> 'misc:Recommends' not in substvars
        True
        >>> substvars.add_dependency('misc:Recommends', "foo (>= 1.0)")
        >>> substvars['misc:Recommends']
        'foo (>= 1.0)'
        >>> # It can be appended to other variables
        >>> substvars['foo'] = 'bar, golf'
        >>> substvars.add_dependency('foo', 'dpkg (>= 1.20.0)')
        >>> substvars['foo']
        'bar, dpkg (>= 1.20.0), golf'
        >>> # Exact duplicates are ignored
        >>> substvars.add_dependency('foo', 'dpkg (>= 1.20.0)')
        >>> substvars['foo']
        'bar, dpkg (>= 1.20.0), golf'

        """
    try:
        variable = self._vars[substvar]
    except KeyError:
        variable = Substvar()
        self._vars[substvar] = variable
    variable.add_dependency(dependency_clause)