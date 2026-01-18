import logging
import os
import sys
from typing import Optional
import wandb
from ..lib import tracelog
from . import _startup_debug, port_file
from .server_sock import SocketServer
from .streams import StreamMux
def _startup_debug_print(self, message: str) -> None:
    if not self._startup_debug_enabled:
        return
    _startup_debug.print_message(message)