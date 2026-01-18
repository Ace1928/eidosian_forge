from __future__ import annotations
import contextlib
import os
import subprocess
import sys
import warnings
from contextlib import ExitStack
from functools import partial
from typing import TYPE_CHECKING, Final, Literal, Protocol, Union, overload
import trio
from ._core import ClosedResourceError, TaskStatus
from ._highlevel_generic import StapledStream
from ._subprocess_platform import (
from ._sync import Lock
from ._util import NoPublicConstructor, final
def _close_pidfd(self) -> None:
    if self._pidfd is not None:
        trio.lowlevel.notify_closing(self._pidfd.fileno())
        self._pidfd.close()
        self._pidfd = None