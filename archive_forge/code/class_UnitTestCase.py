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
class UnitTestCase(Class):
    nofuncargs = True

    def collect(self) -> Iterable[Union[Item, Collector]]:
        from unittest import TestLoader
        cls = self.obj
        if not getattr(cls, '__test__', True):
            return
        skipped = _is_skipped(cls)
        if not skipped:
            self._register_unittest_setup_method_fixture(cls)
            self._register_unittest_setup_class_fixture(cls)
            self._register_setup_class_fixture()
        self.session._fixturemanager.parsefactories(self, unittest=True)
        loader = TestLoader()
        foundsomething = False
        for name in loader.getTestCaseNames(self.obj):
            x = getattr(self.obj, name)
            if not getattr(x, '__test__', True):
                continue
            funcobj = getimfunc(x)
            yield TestCaseFunction.from_parent(self, name=name, callobj=funcobj)
            foundsomething = True
        if not foundsomething:
            runtest = getattr(self.obj, 'runTest', None)
            if runtest is not None:
                ut = sys.modules.get('twisted.trial.unittest', None)
                if ut is None or runtest != ut.TestCase.runTest:
                    yield TestCaseFunction.from_parent(self, name='runTest')

    def _register_unittest_setup_class_fixture(self, cls: type) -> None:
        """Register an auto-use fixture to invoke setUpClass and
        tearDownClass (#517)."""
        setup = getattr(cls, 'setUpClass', None)
        teardown = getattr(cls, 'tearDownClass', None)
        if setup is None and teardown is None:
            return None
        cleanup = getattr(cls, 'doClassCleanups', lambda: None)

        def unittest_setup_class_fixture(request: FixtureRequest) -> Generator[None, None, None]:
            cls = request.cls
            if _is_skipped(cls):
                reason = cls.__unittest_skip_why__
                raise pytest.skip.Exception(reason, _use_item_location=True)
            if setup is not None:
                try:
                    setup()
                except Exception:
                    cleanup()
                    raise
            yield
            try:
                if teardown is not None:
                    teardown()
            finally:
                cleanup()
        self.session._fixturemanager._register_fixture(name=f'_unittest_setUpClass_fixture_{cls.__qualname__}', func=unittest_setup_class_fixture, nodeid=self.nodeid, scope='class', autouse=True)

    def _register_unittest_setup_method_fixture(self, cls: type) -> None:
        """Register an auto-use fixture to invoke setup_method and
        teardown_method (#517)."""
        setup = getattr(cls, 'setup_method', None)
        teardown = getattr(cls, 'teardown_method', None)
        if setup is None and teardown is None:
            return None

        def unittest_setup_method_fixture(request: FixtureRequest) -> Generator[None, None, None]:
            self = request.instance
            if _is_skipped(self):
                reason = self.__unittest_skip_why__
                raise pytest.skip.Exception(reason, _use_item_location=True)
            if setup is not None:
                setup(self, request.function)
            yield
            if teardown is not None:
                teardown(self, request.function)
        self.session._fixturemanager._register_fixture(name=f'_unittest_setup_method_fixture_{cls.__qualname__}', func=unittest_setup_method_fixture, nodeid=self.nodeid, scope='function', autouse=True)