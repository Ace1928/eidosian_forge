from __future__ import annotations
import errno
import glob
import json
import os
import socket
import stat
import tempfile
import warnings
from getpass import getpass
from typing import TYPE_CHECKING, Any, Dict, Union, cast
import zmq
from jupyter_core.paths import jupyter_data_dir, jupyter_runtime_dir, secure_write
from traitlets import Bool, CaselessStrEnum, Instance, Integer, Type, Unicode, observe
from traitlets.config import LoggingConfigurable, SingletonConfigurable
from .localinterfaces import localhost
from .utils import _filefind
class LocalPortCache(SingletonConfigurable):
    """
    Used to keep track of local ports in order to prevent race conditions that
    can occur between port acquisition and usage by the kernel.  All locally-
    provisioned kernels should use this mechanism to limit the possibility of
    race conditions.  Note that this does not preclude other applications from
    acquiring a cached but unused port, thereby re-introducing the issue this
    class is attempting to resolve (minimize).
    See: https://github.com/jupyter/jupyter_client/issues/487
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.currently_used_ports: set[int] = set()

    def find_available_port(self, ip: str) -> int:
        while True:
            tmp_sock = socket.socket()
            tmp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b'\x00' * 8)
            tmp_sock.bind((ip, 0))
            port = tmp_sock.getsockname()[1]
            tmp_sock.close()
            if port not in self.currently_used_ports:
                self.currently_used_ports.add(port)
                return port

    def return_port(self, port: int) -> None:
        if port in self.currently_used_ports:
            self.currently_used_ports.remove(port)