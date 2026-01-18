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
def _reconcile_connection_info(self, info: KernelConnectionInfo) -> None:
    """Reconciles the connection information returned from the Provisioner.

        Because some provisioners (like derivations of LocalProvisioner) may have already
        written the connection file, this method needs to ensure that, if the connection
        file exists, its contents match that of what was returned by the provisioner.  If
        the file does exist and its contents do not match, the file will be replaced with
        the provisioner information (which is considered the truth).

        If the file does not exist, the connection information in 'info' is loaded into the
        KernelManager and written to the file.
        """
    file_exists: bool = False
    if os.path.exists(self.connection_file):
        with open(self.connection_file) as f:
            file_info = json.load(f)
        file_info['key'] = file_info['key'].encode()
        if not self._equal_connections(info, file_info):
            os.remove(self.connection_file)
            self._connection_file_written = False
        else:
            file_exists = True
    if not file_exists:
        for name in port_names:
            setattr(self, name, 0)
        self.load_connection_info(info)
        self.write_connection_file()
    km_info = self.get_connection_info()
    if not self._equal_connections(info, km_info):
        msg = "KernelManager's connection information already exists and does not match the expected values returned from provisioner!"
        raise ValueError(msg)