import contextlib
import itertools
import logging
import sys
import time
from typing import IO, Generator, Optional
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.logging import get_indentation
class InteractiveSpinner(SpinnerInterface):

    def __init__(self, message: str, file: Optional[IO[str]]=None, spin_chars: str='-\\|/', min_update_interval_seconds: float=0.125):
        self._message = message
        if file is None:
            file = sys.stdout
        self._file = file
        self._rate_limiter = RateLimiter(min_update_interval_seconds)
        self._finished = False
        self._spin_cycle = itertools.cycle(spin_chars)
        self._file.write(' ' * get_indentation() + self._message + ' ... ')
        self._width = 0

    def _write(self, status: str) -> None:
        assert not self._finished
        backup = '\x08' * self._width
        self._file.write(backup + ' ' * self._width + backup)
        self._file.write(status)
        self._width = len(status)
        self._file.flush()
        self._rate_limiter.reset()

    def spin(self) -> None:
        if self._finished:
            return
        if not self._rate_limiter.ready():
            return
        self._write(next(self._spin_cycle))

    def finish(self, final_status: str) -> None:
        if self._finished:
            return
        self._write(final_status)
        self._file.write('\n')
        self._file.flush()
        self._finished = True