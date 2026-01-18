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
def _postmortem_traceback(excinfo: ExceptionInfo[BaseException]) -> types.TracebackType:
    from doctest import UnexpectedException
    if isinstance(excinfo.value, UnexpectedException):
        return excinfo.value.exc_info[2]
    elif isinstance(excinfo.value, ConftestImportFailure):
        assert excinfo.value.cause.__traceback__ is not None
        return excinfo.value.cause.__traceback__
    else:
        assert excinfo._excinfo is not None
        return excinfo._excinfo[2]