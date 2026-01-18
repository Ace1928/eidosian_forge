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
class TestCaseFunction(Function):
    nofuncargs = True
    _excinfo: Optional[List[_pytest._code.ExceptionInfo[BaseException]]] = None
    _testcase: Optional['unittest.TestCase'] = None

    def _getobj(self):
        assert self.parent is not None
        return getattr(self.parent.obj, self.originalname)

    def setup(self) -> None:
        self._explicit_tearDown: Optional[Callable[[], None]] = None
        assert self.parent is not None
        self._testcase = self.parent.obj(self.name)
        self._obj = getattr(self._testcase, self.name)
        super().setup()

    def teardown(self) -> None:
        super().teardown()
        if self._explicit_tearDown is not None:
            self._explicit_tearDown()
            self._explicit_tearDown = None
        self._testcase = None
        self._obj = None

    def startTest(self, testcase: 'unittest.TestCase') -> None:
        pass

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

    def addError(self, testcase: 'unittest.TestCase', rawexcinfo: '_SysExcInfoType') -> None:
        try:
            if isinstance(rawexcinfo[1], exit.Exception):
                exit(rawexcinfo[1].msg)
        except TypeError:
            pass
        self._addexcinfo(rawexcinfo)

    def addFailure(self, testcase: 'unittest.TestCase', rawexcinfo: '_SysExcInfoType') -> None:
        self._addexcinfo(rawexcinfo)

    def addSkip(self, testcase: 'unittest.TestCase', reason: str) -> None:
        try:
            raise pytest.skip.Exception(reason, _use_item_location=True)
        except skip.Exception:
            self._addexcinfo(sys.exc_info())

    def addExpectedFailure(self, testcase: 'unittest.TestCase', rawexcinfo: '_SysExcInfoType', reason: str='') -> None:
        try:
            xfail(str(reason))
        except xfail.Exception:
            self._addexcinfo(sys.exc_info())

    def addUnexpectedSuccess(self, testcase: 'unittest.TestCase', reason: Optional['twisted.trial.unittest.Todo']=None) -> None:
        msg = 'Unexpected success'
        if reason:
            msg += f': {reason.reason}'
        try:
            fail(msg, pytrace=False)
        except fail.Exception:
            self._addexcinfo(sys.exc_info())

    def addSuccess(self, testcase: 'unittest.TestCase') -> None:
        pass

    def stopTest(self, testcase: 'unittest.TestCase') -> None:
        pass

    def addDuration(self, testcase: 'unittest.TestCase', elapsed: float) -> None:
        pass

    def runtest(self) -> None:
        from _pytest.debugging import maybe_wrap_pytest_function_for_tracing
        assert self._testcase is not None
        maybe_wrap_pytest_function_for_tracing(self)
        if is_async_function(self.obj):
            self._testcase(result=self)
        else:
            assert isinstance(self.parent, UnitTestCase)
            skipped = _is_skipped(self.obj) or _is_skipped(self.parent.obj)
            if self.config.getoption('usepdb') and (not skipped):
                self._explicit_tearDown = self._testcase.tearDown
                setattr(self._testcase, 'tearDown', lambda *args: None)
            setattr(self._testcase, self.name, self.obj)
            try:
                self._testcase(result=self)
            finally:
                delattr(self._testcase, self.name)

    def _traceback_filter(self, excinfo: _pytest._code.ExceptionInfo[BaseException]) -> _pytest._code.Traceback:
        traceback = super()._traceback_filter(excinfo)
        ntraceback = traceback.filter(lambda x: not x.frame.f_globals.get('__unittest'))
        if not ntraceback:
            ntraceback = traceback
        return ntraceback