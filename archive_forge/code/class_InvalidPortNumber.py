import asyncio
import atexit
import time
import typing as t
from queue import Empty
from threading import Event, Thread
import zmq.asyncio
from jupyter_core.utils import ensure_async
from ._version import protocol_version_info
from .channelsabc import HBChannelABC
from .session import Session
class InvalidPortNumber(Exception):
    """An exception raised for an invalid port number."""
    pass