import abc
import collections
import contextlib
import io
from io import UnsupportedOperation
import os
import sys
from tempfile import TemporaryFile
from types import TracebackType
from typing import Any
from typing import AnyStr
from typing import BinaryIO
from typing import Final
from typing import final
from typing import Generator
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Literal
from typing import NamedTuple
from typing import Optional
from typing import TextIO
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from _pytest.config import Config
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import SubRequest
from _pytest.nodes import Collector
from _pytest.nodes import File
from _pytest.nodes import Item
from _pytest.reports import CollectReport
def _get_multicapture(method: _CaptureMethod) -> MultiCapture[str]:
    if method == 'fd':
        return MultiCapture(in_=FDCapture(0), out=FDCapture(1), err=FDCapture(2))
    elif method == 'sys':
        return MultiCapture(in_=SysCapture(0), out=SysCapture(1), err=SysCapture(2))
    elif method == 'no':
        return MultiCapture(in_=None, out=None, err=None)
    elif method == 'tee-sys':
        return MultiCapture(in_=None, out=SysCapture(1, tee=True), err=SysCapture(2, tee=True))
    raise ValueError(f'unknown capturing method: {method!r}')