from datetime import datetime
import functools
import os
import platform
import re
from typing import Callable
from typing import Dict
from typing import List
from typing import Match
from typing import Optional
from typing import Tuple
from typing import Union
import xml.etree.ElementTree as ET
from _pytest import nodes
from _pytest import timing
from _pytest._code.code import ExceptionRepr
from _pytest._code.code import ReprFileLocation
from _pytest.config import Config
from _pytest.config import filename_arg
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.reports import TestReport
from _pytest.stash import StashKey
from _pytest.terminal import TerminalReporter
import pytest
def _check_record_param_type(param: str, v: str) -> None:
    """Used by record_testsuite_property to check that the given parameter name is of the proper
    type."""
    __tracebackhide__ = True
    if not isinstance(v, str):
        msg = '{param} parameter needs to be a string, but {g} given'
        raise TypeError(msg.format(param=param, g=type(v).__name__))