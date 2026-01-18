import sys
from collections import OrderedDict
from contextlib import asynccontextmanager
from functools import partial
from ipaddress import ip_address
import itertools
import logging
import random
import ssl
import struct
import urllib.parse
from typing import List, Optional, Union
import trio
import trio.abc
from wsproto import ConnectionType, WSConnection
from wsproto.connection import ConnectionState
import wsproto.frame_protocol as wsframeproto
from wsproto.events import (
import wsproto.utilities
class ConnectionRejected(HandshakeError):
    """
    A WebSocket connection could not be established because the server rejected
    the connection attempt.
    """

    def __init__(self, status_code, headers, body):
        """
        Constructor.

        :param reason:
        :type reason: CloseReason
        """
        super().__init__()
        self.status_code = status_code
        self.headers = headers
        self.body = body

    def __repr__(self):
        """ Return representation. """
        return f'{self.__class__.__name__}<status_code={self.status_code}>'