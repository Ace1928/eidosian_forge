import sys
import traceback
import types
from typing import Any
from typing import Callable
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import _pytest._code
from _pytest.compat import getimfunc
from _pytest.compat import is_async_function
from _pytest.config import hookimpl
from _pytest.fixtures import FixtureRequest
from _pytest.nodes import Collector
from _pytest.nodes import Item
from _pytest.outcomes import exit
from _pytest.outcomes import fail
from _pytest.outcomes import skip
from _pytest.outcomes import xfail
from _pytest.python import Class
from _pytest.python import Function
from _pytest.python import Module
from _pytest.runner import CallInfo
import pytest
def _addexcinfo(self, rawexcinfo: '_SysExcInfoType') -> None:
    rawexcinfo = getattr(rawexcinfo, '_rawexcinfo', rawexcinfo)
    try:
        excinfo = _pytest._code.ExceptionInfo[BaseException].from_exc_info(rawexcinfo)
        _ = excinfo.value
        _ = excinfo.traceback
    except TypeError:
        try:
            try:
                values = traceback.format_exception(*rawexcinfo)
                values.insert(0, 'NOTE: Incompatible Exception Representation, displaying natively:\n\n')
                fail(''.join(values), pytrace=False)
            except (fail.Exception, KeyboardInterrupt):
                raise
            except BaseException:
                fail(f'ERROR: Unknown Incompatible Exception representation:\n{rawexcinfo!r}', pytrace=False)
        except KeyboardInterrupt:
            raise
        except fail.Exception:
            excinfo = _pytest._code.ExceptionInfo.from_current()
    self.__dict__.setdefault('_excinfo', []).append(excinfo)