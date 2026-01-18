from __future__ import annotations
from typing import Callable
from typing_extensions import NoReturn, Protocol
from twisted.python import randbytes
from twisted.trial import unittest
class _SupportsAssertions(Protocol):

    def assertEqual(self, a: object, b: object) -> object:
        ...

    def assertNotEqual(self, a: object, b: object) -> object:
        ...