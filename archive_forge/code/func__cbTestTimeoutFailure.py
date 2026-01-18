from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
def _cbTestTimeoutFailure(self, res: failure.Failure) -> None:
    self.assertTrue(hasattr(res, 'type'))
    self.assertEqual(res.type, helper.ExpectationTimeout)