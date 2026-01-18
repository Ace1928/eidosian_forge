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
def _showfixtures_main(config: Config, session: Session) -> None:
    import _pytest.config
    session.perform_collect()
    invocation_dir = config.invocation_params.dir
    tw = _pytest.config.create_terminal_writer(config)
    verbose = config.getvalue('verbose')
    fm = session._fixturemanager
    available = []
    seen: Set[Tuple[str, str]] = set()
    for argname, fixturedefs in fm._arg2fixturedefs.items():
        assert fixturedefs is not None
        if not fixturedefs:
            continue
        for fixturedef in fixturedefs:
            loc = getlocation(fixturedef.func, invocation_dir)
            if (fixturedef.argname, loc) in seen:
                continue
            seen.add((fixturedef.argname, loc))
            available.append((len(fixturedef.baseid), fixturedef.func.__module__, _pretty_fixture_path(invocation_dir, fixturedef.func), fixturedef.argname, fixturedef))
    available.sort()
    currentmodule = None
    for baseid, module, prettypath, argname, fixturedef in available:
        if currentmodule != module:
            if not module.startswith('_pytest.'):
                tw.line()
                tw.sep('-', f'fixtures defined from {module}')
                currentmodule = module
        if verbose <= 0 and argname.startswith('_'):
            continue
        tw.write(f'{argname}', green=True)
        if fixturedef.scope != 'function':
            tw.write(' [%s scope]' % fixturedef.scope, cyan=True)
        tw.write(f' -- {prettypath}', yellow=True)
        tw.write('\n')
        doc = inspect.getdoc(fixturedef.func)
        if doc:
            write_docstring(tw, doc.split('\n\n')[0] if verbose <= 0 else doc)
        else:
            tw.line('    no docstring available', red=True)
        tw.line()