import inspect
import selectors
import socket
import threading
import time
from typing import Any, Callable, Optional, Union
from . import _logging
from ._abnf import ABNF
from ._core import WebSocket, getdefaulttimeout
from ._exceptions import (
from ._url import parse_url
def _stop_ping_thread(self) -> None:
    if self.stop_ping:
        self.stop_ping.set()
    if self.ping_thread and self.ping_thread.is_alive():
        self.ping_thread.join(3)
    self.last_ping_tm = self.last_pong_tm = float(0)