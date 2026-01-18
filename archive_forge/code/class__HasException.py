from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
class _HasException(Protocol):

    @property
    def exception(self) -> BaseException:
        ...