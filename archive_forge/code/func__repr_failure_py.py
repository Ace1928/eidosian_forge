import abc
from functools import cached_property
from inspect import signature
import os
import pathlib
from pathlib import Path
from typing import Any
from typing import Callable
from typing import cast
from typing import Iterable
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings
import pluggy
import _pytest._code
from _pytest._code import getfslineno
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import TerminalRepr
from _pytest._code.code import Traceback
from _pytest.compat import LEGACY_PATH
from _pytest.config import Config
from _pytest.config import ConftestImportFailure
from _pytest.config.compat import _check_path
from _pytest.deprecated import NODE_CTOR_FSPATH_ARG
from _pytest.mark.structures import Mark
from _pytest.mark.structures import MarkDecorator
from _pytest.mark.structures import NodeKeywords
from _pytest.outcomes import fail
from _pytest.pathlib import absolutepath
from _pytest.pathlib import commonpath
from _pytest.stash import Stash
from _pytest.warning_types import PytestWarning
def _repr_failure_py(self, excinfo: ExceptionInfo[BaseException], style: 'Optional[_TracebackStyle]'=None) -> TerminalRepr:
    from _pytest.fixtures import FixtureLookupError
    if isinstance(excinfo.value, ConftestImportFailure):
        excinfo = ExceptionInfo.from_exception(excinfo.value.cause)
    if isinstance(excinfo.value, fail.Exception):
        if not excinfo.value.pytrace:
            style = 'value'
    if isinstance(excinfo.value, FixtureLookupError):
        return excinfo.value.formatrepr()
    tbfilter: Union[bool, Callable[[ExceptionInfo[BaseException]], Traceback]]
    if self.config.getoption('fulltrace', False):
        style = 'long'
        tbfilter = False
    else:
        tbfilter = self._traceback_filter
        if style == 'auto':
            style = 'long'
    if style is None:
        if self.config.getoption('tbstyle', 'auto') == 'short':
            style = 'short'
        else:
            style = 'long'
    if self.config.getoption('verbose', 0) > 1:
        truncate_locals = False
    else:
        truncate_locals = True
    try:
        abspath = Path(os.getcwd()) != self.config.invocation_params.dir
    except OSError:
        abspath = True
    return excinfo.getrepr(funcargs=True, abspath=abspath, showlocals=self.config.getoption('showlocals', False), style=style, tbfilter=tbfilter, truncate_locals=truncate_locals)