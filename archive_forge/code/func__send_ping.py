import copy
import os
import socket
import ssl
import sys
import threading
import weakref
from abc import abstractmethod
from itertools import chain
from queue import Empty, Full, LifoQueue
from time import time
from typing import Any, Callable, List, Optional, Type, Union
from urllib.parse import parse_qs, unquote, urlparse
from ._parsers import Encoder, _HiredisParser, _RESP2Parser, _RESP3Parser
from .backoff import NoBackoff
from .credentials import CredentialProvider, UsernamePasswordCredentialProvider
from .exceptions import (
from .retry import Retry
from .utils import (
def _send_ping(self):
    """Send PING, expect PONG in return"""
    self.send_command('PING', check_health=False)
    if str_if_bytes(self.read_response()) != 'PONG':
        raise ConnectionError('Bad response from PING health check')