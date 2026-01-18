import argparse
import functools
import sys
import types
from typing import Any
from typing import Callable
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import unittest
from _pytest import outcomes
from _pytest._code import ExceptionInfo
from _pytest.config import Config
from _pytest.config import ConftestImportFailure
from _pytest.config import hookimpl
from _pytest.config import PytestPluginManager
from _pytest.config.argparsing import Parser
from _pytest.config.exceptions import UsageError
from _pytest.nodes import Node
from _pytest.reports import BaseReport
class PdbInvoke:

    def pytest_exception_interact(self, node: Node, call: 'CallInfo[Any]', report: BaseReport) -> None:
        capman = node.config.pluginmanager.getplugin('capturemanager')
        if capman:
            capman.suspend_global_capture(in_=True)
            out, err = capman.read_global_capture()
            sys.stdout.write(out)
            sys.stdout.write(err)
        assert call.excinfo is not None
        if not isinstance(call.excinfo.value, unittest.SkipTest):
            _enter_pdb(node, call.excinfo, report)

    def pytest_internalerror(self, excinfo: ExceptionInfo[BaseException]) -> None:
        tb = _postmortem_traceback(excinfo)
        post_mortem(tb)