import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def _applicationDataTest(self, data, calls):
    occs = occurrences(self.proto)
    self.parser.dataReceived(data)
    while calls:
        self.assertCall(occs.pop(0), *calls.pop(0))
    self.assertFalse(occs, f'No other calls should happen: {occs!r}')