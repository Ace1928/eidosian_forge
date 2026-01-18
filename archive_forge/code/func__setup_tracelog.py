import logging
import os
import sys
from typing import Optional
import wandb
from ..lib import tracelog
from . import _startup_debug, port_file
from .server_sock import SocketServer
from .streams import StreamMux
def _setup_tracelog(self) -> None:
    tracelog_mode = os.environ.get('WANDB_TRACELOG')
    if tracelog_mode:
        tracelog.enable(tracelog_mode)