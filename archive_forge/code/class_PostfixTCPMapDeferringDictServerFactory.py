from __future__ import annotations
import sys
from collections import UserDict
from typing import TYPE_CHECKING, Union
from urllib.parse import quote as _quote, unquote as _unquote
from twisted.internet import defer, protocol
from twisted.protocols import basic, policies
from twisted.python import log
class PostfixTCPMapDeferringDictServerFactory(protocol.ServerFactory):
    """
    An in-memory dictionary factory for PostfixTCPMapServer.
    """
    protocol = PostfixTCPMapServer

    def __init__(self, data=None):
        self.data = {}
        if data is not None:
            self.data.update(data)

    def get(self, key):
        return defer.succeed(self.data.get(key))