import os
from typing import Optional, Union
from warnings import warn
from zope.interface import implementer
import attr
from typing_extensions import Literal
from twisted.internet.interfaces import IAddress
from twisted.python.filepath import _asFilesystemBytes, _coerceToFilesystemEncoding
from twisted.python.runtime import platform
@attr.s(hash=True, auto_attribs=True)
@implementer(IAddress)
class HostnameAddress:
    """
    A L{HostnameAddress} represents the address of a L{HostnameEndpoint}.

    @ivar hostname: A hostname byte string; for example, b"example.com".
    @type hostname: L{bytes}

    @ivar port: An integer representing the port number.
    @type port: L{int}
    """
    hostname: bytes
    port: int