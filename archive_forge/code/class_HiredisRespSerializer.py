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
class HiredisRespSerializer:

    def pack(self, *args: List):
        """Pack a series of arguments into the Redis protocol"""
        output = []
        if isinstance(args[0], str):
            args = tuple(args[0].encode().split()) + args[1:]
        elif b' ' in args[0]:
            args = tuple(args[0].split()) + args[1:]
        try:
            output.append(hiredis.pack_command(args))
        except TypeError:
            _, value, traceback = sys.exc_info()
            raise DataError(value).with_traceback(traceback)
        return output