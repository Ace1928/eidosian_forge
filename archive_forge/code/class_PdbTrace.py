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
class PdbTrace:

    @hookimpl(wrapper=True)
    def pytest_pyfunc_call(self, pyfuncitem) -> Generator[None, object, object]:
        wrap_pytest_function_for_tracing(pyfuncitem)
        return (yield)