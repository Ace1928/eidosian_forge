import bdb
import builtins
import inspect
import os
import platform
import sys
import traceback
import types
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import (
import pytest
from _pytest import outcomes
from _pytest._code.code import ExceptionInfo, ReprFileLocation, TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.compat import safe_getattr
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.nodes import Collector
from _pytest.outcomes import OutcomeException
from _pytest.pathlib import fnmatch_ex, import_path
from _pytest.python_api import approx
from _pytest.warning_types import PytestWarning
class IPDoctestItem(pytest.Item):
    _user_ns_orig: Dict[str, Any]

    def __init__(self, name: str, parent: 'Union[IPDoctestTextfile, IPDoctestModule]', runner: Optional['IPDocTestRunner']=None, dtest: Optional['doctest.DocTest']=None) -> None:
        super().__init__(name, parent)
        self.runner = runner
        self.dtest = dtest
        self.obj = None
        self.fixture_request: Optional[FixtureRequest] = None
        self._user_ns_orig = {}

    @classmethod
    def from_parent(cls, parent: 'Union[IPDoctestTextfile, IPDoctestModule]', *, name: str, runner: 'IPDocTestRunner', dtest: 'doctest.DocTest'):
        """The public named constructor."""
        return super().from_parent(name=name, parent=parent, runner=runner, dtest=dtest)

    def setup(self) -> None:
        if self.dtest is not None:
            self.fixture_request = _setup_fixtures(self)
            globs = dict(getfixture=self.fixture_request.getfixturevalue)
            for name, value in self.fixture_request.getfixturevalue('ipdoctest_namespace').items():
                globs[name] = value
            self.dtest.globs.update(globs)
            from .ipdoctest import IPExample
            if isinstance(self.dtest.examples[0], IPExample):
                self._user_ns_orig = {}
                self._user_ns_orig.update(_ip.user_ns)
                _ip.user_ns.update(self.dtest.globs)
                _ip.user_ns.pop('_', None)
                _ip.user_ns['__builtins__'] = builtins
                self.dtest.globs = _ip.user_ns

    def teardown(self) -> None:
        from .ipdoctest import IPExample
        if isinstance(self.dtest.examples[0], IPExample):
            self.dtest.globs = {}
            _ip.user_ns.clear()
            _ip.user_ns.update(self._user_ns_orig)
            del self._user_ns_orig
        self.dtest.globs.clear()

    def runtest(self) -> None:
        assert self.dtest is not None
        assert self.runner is not None
        _check_all_skipped(self.dtest)
        self._disable_output_capturing_for_darwin()
        failures: List['doctest.DocTestFailure'] = []
        had_underscore_value = hasattr(builtins, '_')
        underscore_original_value = getattr(builtins, '_', None)
        curdir = os.getcwd()
        os.chdir(self.fspath.dirname)
        try:
            self.runner.run(self.dtest, out=failures, clear_globs=False)
        finally:
            os.chdir(curdir)
            if had_underscore_value:
                setattr(builtins, '_', underscore_original_value)
            elif hasattr(builtins, '_'):
                delattr(builtins, '_')
        if failures:
            raise MultipleDoctestFailures(failures)

    def _disable_output_capturing_for_darwin(self) -> None:
        """Disable output capturing. Otherwise, stdout is lost to ipdoctest (pytest#985)."""
        if platform.system() != 'Darwin':
            return
        capman = self.config.pluginmanager.getplugin('capturemanager')
        if capman:
            capman.suspend_global_capture(in_=True)
            out, err = capman.read_global_capture()
            sys.stdout.write(out)
            sys.stderr.write(err)

    def repr_failure(self, excinfo: ExceptionInfo[BaseException]) -> Union[str, TerminalRepr]:
        import doctest
        failures: Optional[Sequence[Union[doctest.DocTestFailure, doctest.UnexpectedException]]] = None
        if isinstance(excinfo.value, (doctest.DocTestFailure, doctest.UnexpectedException)):
            failures = [excinfo.value]
        elif isinstance(excinfo.value, MultipleDoctestFailures):
            failures = excinfo.value.failures
        if failures is None:
            return super().repr_failure(excinfo)
        reprlocation_lines = []
        for failure in failures:
            example = failure.example
            test = failure.test
            filename = test.filename
            if test.lineno is None:
                lineno = None
            else:
                lineno = test.lineno + example.lineno + 1
            message = type(failure).__name__
            reprlocation = ReprFileLocation(filename, lineno, message)
            checker = _get_checker()
            report_choice = _get_report_choice(self.config.getoption('ipdoctestreport'))
            if lineno is not None:
                assert failure.test.docstring is not None
                lines = failure.test.docstring.splitlines(False)
                assert test.lineno is not None
                lines = ['%03d %s' % (i + test.lineno + 1, x) for i, x in enumerate(lines)]
                lines = lines[max(example.lineno - 9, 0):example.lineno + 1]
            else:
                lines = ['EXAMPLE LOCATION UNKNOWN, not showing all tests of that example']
                indent = '>>>'
                for line in example.source.splitlines():
                    lines.append(f'??? {indent} {line}')
                    indent = '...'
            if isinstance(failure, doctest.DocTestFailure):
                lines += checker.output_difference(example, failure.got, report_choice).split('\n')
            else:
                inner_excinfo = ExceptionInfo.from_exc_info(failure.exc_info)
                lines += ['UNEXPECTED EXCEPTION: %s' % repr(inner_excinfo.value)]
                lines += [x.strip('\n') for x in traceback.format_exception(*failure.exc_info)]
            reprlocation_lines.append((reprlocation, lines))
        return ReprFailDoctest(reprlocation_lines)

    def reportinfo(self) -> Tuple[Union['os.PathLike[str]', str], Optional[int], str]:
        assert self.dtest is not None
        return (self.path, self.dtest.lineno, '[ipdoctest] %s' % self.name)
    if int(pytest.__version__.split('.')[0]) < 7:

        @property
        def path(self) -> Path:
            return Path(self.fspath)