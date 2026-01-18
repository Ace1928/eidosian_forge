from __future__ import annotations
import copyreg
import io
import pickle
import sys
import textwrap
from typing import Any, Callable, List, Tuple
from typing_extensions import NoReturn
from twisted.persisted import aot, crefutil, styles
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
class NonDictState:
    state: str

    def __getstate__(self) -> str:
        return self.state

    def __setstate__(self, state: str) -> None:
        self.state = state