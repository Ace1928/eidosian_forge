from __future__ import annotations
import weakref
from typing import TYPE_CHECKING, Callable, Union
import pytest
from .. import _core
from .._sync import *
from .._timeouts import sleep_forever
from ..testing import assert_checkpoints, wait_all_tasks_blocked
from .._channel import open_memory_channel
from .._sync import AsyncContextManagerMixin
def acquire_nowait(self) -> None:
    assert not self.acquired
    self.acquired = True