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
class PicklingTests(TestCase):
    """Test pickling of extra object types."""

    def test_module(self) -> None:
        pickl = pickle.dumps(styles)
        o = pickle.loads(pickl)
        self.assertEqual(o, styles)

    def test_instanceMethod(self) -> None:
        obj = Pickleable(4)
        pickl = pickle.dumps(obj.getX)
        o = pickle.loads(pickl)
        self.assertEqual(o(), 4)
        self.assertEqual(type(o), type(obj.getX))