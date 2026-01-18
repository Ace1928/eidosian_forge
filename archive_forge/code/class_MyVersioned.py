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
class MyVersioned(styles.Versioned):
    persistenceVersion = 2
    persistenceForgets = ['garbagedata']
    v3 = 0
    v4 = 0

    def __init__(self) -> None:
        self.somedata = 'xxx'
        self.garbagedata = lambda q: 'cant persist'

    def upgradeToVersion3(self) -> None:
        self.v3 += 1

    def upgradeToVersion4(self) -> None:
        self.v4 += 1