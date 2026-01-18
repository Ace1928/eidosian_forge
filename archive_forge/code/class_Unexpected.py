from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
class Unexpected:

    def __str__(self) -> str:
        return 'wrong'

    def __repr__(self) -> str:
        return '<unexpected>'