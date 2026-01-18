import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def _makeControlFunctionSymbols(name, colOffset, names, doc):
    attrs = {name: ValueConstant(_ecmaCodeTableCoordinate(i + colOffset, j)) for j, row in enumerate(names) for i, name in enumerate(row) if name}
    attrs['__doc__'] = doc
    return type(name, (Values,), attrs)