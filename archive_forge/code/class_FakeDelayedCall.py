from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
class FakeDelayedCall:
    called = False
    cancelled = False

    def __init__(self, fs: FakeScheduler, timeout: float, f: Callable[..., None], a: tuple[object, ...], kw: dict[str, object]) -> None:
        self.fs = fs
        self.timeout = timeout
        self.f = f
        self.a = a
        self.kw = kw

    def active(self) -> bool:
        return not (self.cancelled or self.called)

    def cancel(self) -> None:
        self.cancelled = True

    def call(self) -> None:
        self.called = True
        self.f(*self.a, **self.kw)