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
def _start_ping_thread(self) -> None:
    self.last_ping_tm = self.last_pong_tm = float(0)
    self.stop_ping = threading.Event()
    self.ping_thread = threading.Thread(target=self._send_ping)
    self.ping_thread.daemon = True
    self.ping_thread.start()