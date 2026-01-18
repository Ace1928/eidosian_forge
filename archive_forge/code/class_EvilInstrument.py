from __future__ import annotations
from typing import TYPE_CHECKING, Container, Iterable, NoReturn
import attrs
import pytest
from ... import _abc, _core
from .tutil import check_sequence_matches
class EvilInstrument(_abc.Instrument):

    def task_exited(self, task: Task) -> NoReturn:
        raise AssertionError('this should never happen')

    @property
    def after_run(self) -> NoReturn:
        raise ValueError('oops')