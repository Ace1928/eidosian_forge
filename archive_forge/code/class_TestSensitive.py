from __future__ import annotations
import os
import sys
import types
from typing_extensions import NoReturn
from twisted.python import rebuild
from twisted.trial.unittest import TestCase
from . import crash_test_dummy
class TestSensitive(Sensitive):

    def test_method(self) -> None:
        """
                Dummy method
                """