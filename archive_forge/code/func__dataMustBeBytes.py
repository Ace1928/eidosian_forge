from socket import AF_INET, AF_INET6, inet_pton
from typing import Iterable, List, Optional
from zope.interface import implementer
from twisted.internet import interfaces, main
from twisted.python import failure, reflect
from twisted.python.compat import lazyByteSlice
def _dataMustBeBytes(obj):
    if not isinstance(obj, bytes):
        raise TypeError('Data must be bytes')