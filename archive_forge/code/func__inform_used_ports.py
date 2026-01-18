import logging
import os
import sys
from typing import Optional
import wandb
from ..lib import tracelog
from . import _startup_debug, port_file
from .server_sock import SocketServer
from .streams import StreamMux
def _inform_used_ports(self, sock_port: Optional[int]) -> None:
    if not self._port_fname:
        return
    pf = port_file.PortFile(sock_port=sock_port)
    pf.write(self._port_fname)