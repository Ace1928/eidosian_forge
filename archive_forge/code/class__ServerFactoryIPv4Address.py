import os
from typing import Optional, Union
from warnings import warn
from zope.interface import implementer
import attr
from typing_extensions import Literal
from twisted.internet.interfaces import IAddress
from twisted.python.filepath import _asFilesystemBytes, _coerceToFilesystemEncoding
from twisted.python.runtime import platform
class _ServerFactoryIPv4Address(IPv4Address):
    """Backwards compatibility hack. Just like IPv4Address in practice."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, tuple):
            warn('IPv4Address.__getitem__ is deprecated.  Use attributes instead.', category=DeprecationWarning, stacklevel=2)
            return (self.host, self.port) == other
        elif isinstance(other, IPv4Address):
            a = (self.type, self.host, self.port)
            b = (other.type, other.host, other.port)
            return a == b
        return NotImplemented