from __future__ import annotations
import contextlib
import select
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Literal
import attrs
from .. import _core
from ._io_common import wake_all
from ._run import Task, _public
from ._wakeup_socketpair import WakeupSocketpair
def _update_registrations(self, fd: int) -> None:
    waiters = self._registered[fd]
    wanted_flags = 0
    if waiters.read_task is not None:
        wanted_flags |= select.EPOLLIN
    if waiters.write_task is not None:
        wanted_flags |= select.EPOLLOUT
    if wanted_flags != waiters.current_flags:
        try:
            try:
                self._epoll.modify(fd, wanted_flags | select.EPOLLONESHOT)
            except OSError:
                self._epoll.register(fd, wanted_flags | select.EPOLLONESHOT)
            waiters.current_flags = wanted_flags
        except OSError as exc:
            del self._registered[fd]
            wake_all(waiters, exc)
            return
    if not wanted_flags:
        del self._registered[fd]