from __future__ import annotations
import errno
import os
import sys
from typing import TYPE_CHECKING
import trio
from ._abc import Stream
from ._util import ConflictDetector, final
def _raw_close(self) -> None:
    if self.closed:
        return
    fd = self.fd
    self.fd = -1
    os.set_blocking(fd, self._original_is_blocking)
    os.close(fd)