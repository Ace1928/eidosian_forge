from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IAddress(Interface):
    """
    An address, e.g. a TCP C{(host, port)}.

    Default implementations are in L{twisted.internet.address}.
    """