import abc
from collections import Counter
from collections import defaultdict
import dataclasses
import enum
import fnmatch
from functools import partial
import inspect
import itertools
import os
from pathlib import Path
import types
from typing import Any
from typing import Callable
from typing import Dict
from typing import final
from typing import Generator
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import warnings
import _pytest
from _pytest import fixtures
from _pytest import nodes
from _pytest._code import filter_traceback
from _pytest._code import getfslineno
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import TerminalRepr
from _pytest._code.code import Traceback
from _pytest._io import TerminalWriter
from _pytest._io.saferepr import saferepr
from _pytest.compat import ascii_escaped
from _pytest.compat import get_default_arg_names
from _pytest.compat import get_real_func
from _pytest.compat import getimfunc
from _pytest.compat import getlocation
from _pytest.compat import is_async_function
from _pytest.compat import is_generator
from _pytest.compat import LEGACY_PATH
from _pytest.compat import NOTSET
from _pytest.compat import safe_getattr
from _pytest.compat import safe_isclass
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import FixtureDef
from _pytest.fixtures import FixtureRequest
from _pytest.fixtures import FuncFixtureInfo
from _pytest.fixtures import get_scope_node
from _pytest.main import Session
from _pytest.mark import MARK_GEN
from _pytest.mark import ParameterSet
from _pytest.mark.structures import get_unpacked_marks
from _pytest.mark.structures import Mark
from _pytest.mark.structures import MarkDecorator
from _pytest.mark.structures import normalize_mark_list
from _pytest.outcomes import fail
from _pytest.outcomes import skip
from _pytest.pathlib import bestrelpath
from _pytest.pathlib import fnmatch_ex
from _pytest.pathlib import import_path
from _pytest.pathlib import ImportPathMismatchError
from _pytest.pathlib import scandir
from _pytest.scope import _ScopeName
from _pytest.scope import Scope
from _pytest.stash import StashKey
from _pytest.warning_types import PytestCollectionWarning
from _pytest.warning_types import PytestReturnNotNoneWarning
from _pytest.warning_types import PytestUnhandledCoroutineWarning
class PyobjMixin(nodes.Node):
    """this mix-in inherits from Node to carry over the typing information

    as its intended to always mix in before a node
    its position in the mro is unaffected"""
    _ALLOW_MARKERS = True

    @property
    def module(self):
        """Python module object this node was collected from (can be None)."""
        node = self.getparent(Module)
        return node.obj if node is not None else None

    @property
    def cls(self):
        """Python class object this node was collected from (can be None)."""
        node = self.getparent(Class)
        return node.obj if node is not None else None

    @property
    def instance(self):
        """Python instance object the function is bound to.

        Returns None if not a test method, e.g. for a standalone test function,
        a staticmethod, a class or a module.
        """
        node = self.getparent(Function)
        return getattr(node.obj, '__self__', None) if node is not None else None

    @property
    def obj(self):
        """Underlying Python object."""
        obj = getattr(self, '_obj', None)
        if obj is None:
            self._obj = obj = self._getobj()
            if self._ALLOW_MARKERS:
                self.own_markers.extend(get_unpacked_marks(self.obj))
                self.keywords.update(((mark.name, mark) for mark in self.own_markers))
        return obj

    @obj.setter
    def obj(self, value):
        self._obj = value

    def _getobj(self):
        """Get the underlying Python object. May be overwritten by subclasses."""
        assert self.parent is not None
        obj = self.parent.obj
        return getattr(obj, self.name)

    def getmodpath(self, stopatmodule: bool=True, includemodule: bool=False) -> str:
        """Return Python path relative to the containing module."""
        parts = []
        for node in self.iter_parents():
            name = node.name
            if isinstance(node, Module):
                name = os.path.splitext(name)[0]
                if stopatmodule:
                    if includemodule:
                        parts.append(name)
                    break
            parts.append(name)
        parts.reverse()
        return '.'.join(parts)

    def reportinfo(self) -> Tuple[Union['os.PathLike[str]', str], Optional[int], str]:
        path, lineno = getfslineno(self.obj)
        modpath = self.getmodpath()
        return (path, lineno, modpath)