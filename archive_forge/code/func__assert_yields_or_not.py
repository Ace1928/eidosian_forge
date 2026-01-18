from __future__ import annotations
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING
from .. import _core
@contextmanager
def _assert_yields_or_not(expected: bool) -> Generator[None, None, None]:
    """Check if checkpoints are executed in a block of code."""
    __tracebackhide__ = True
    task = _core.current_task()
    orig_cancel = task._cancel_points
    orig_schedule = task._schedule_points
    try:
        yield
        if expected and (task._cancel_points == orig_cancel or task._schedule_points == orig_schedule):
            raise AssertionError('assert_checkpoints block did not yield!')
    finally:
        if not expected and (task._cancel_points != orig_cancel or task._schedule_points != orig_schedule):
            raise AssertionError('assert_no_checkpoints block yielded!')