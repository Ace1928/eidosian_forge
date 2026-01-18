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
class FDCaptureBase(CaptureBase[AnyStr]):

    def __init__(self, targetfd: int) -> None:
        self.targetfd = targetfd
        try:
            os.fstat(targetfd)
        except OSError:
            self.targetfd_invalid: Optional[int] = os.open(os.devnull, os.O_RDWR)
            os.dup2(self.targetfd_invalid, targetfd)
        else:
            self.targetfd_invalid = None
        self.targetfd_save = os.dup(targetfd)
        if targetfd == 0:
            self.tmpfile = open(os.devnull, encoding='utf-8')
            self.syscapture: CaptureBase[str] = SysCapture(targetfd)
        else:
            self.tmpfile = EncodedFile(TemporaryFile(buffering=0), encoding='utf-8', errors='replace', newline='', write_through=True)
            if targetfd in patchsysdict:
                self.syscapture = SysCapture(targetfd, self.tmpfile)
            else:
                self.syscapture = NoCapture(targetfd)
        self._state = 'initialized'

    def __repr__(self) -> str:
        return '<{} {} oldfd={} _state={!r} tmpfile={!r}>'.format(self.__class__.__name__, self.targetfd, self.targetfd_save, self._state, self.tmpfile)

    def _assert_state(self, op: str, states: Tuple[str, ...]) -> None:
        assert self._state in states, 'cannot {} in state {!r}: expected one of {}'.format(op, self._state, ', '.join(states))

    def start(self) -> None:
        """Start capturing on targetfd using memorized tmpfile."""
        self._assert_state('start', ('initialized',))
        os.dup2(self.tmpfile.fileno(), self.targetfd)
        self.syscapture.start()
        self._state = 'started'

    def done(self) -> None:
        """Stop capturing, restore streams, return original capture file,
        seeked to position zero."""
        self._assert_state('done', ('initialized', 'started', 'suspended', 'done'))
        if self._state == 'done':
            return
        os.dup2(self.targetfd_save, self.targetfd)
        os.close(self.targetfd_save)
        if self.targetfd_invalid is not None:
            if self.targetfd_invalid != self.targetfd:
                os.close(self.targetfd)
            os.close(self.targetfd_invalid)
        self.syscapture.done()
        self.tmpfile.close()
        self._state = 'done'

    def suspend(self) -> None:
        self._assert_state('suspend', ('started', 'suspended'))
        if self._state == 'suspended':
            return
        self.syscapture.suspend()
        os.dup2(self.targetfd_save, self.targetfd)
        self._state = 'suspended'

    def resume(self) -> None:
        self._assert_state('resume', ('started', 'suspended'))
        if self._state == 'started':
            return
        self.syscapture.resume()
        os.dup2(self.tmpfile.fileno(), self.targetfd)
        self._state = 'started'