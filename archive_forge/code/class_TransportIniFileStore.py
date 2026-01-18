import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
class TransportIniFileStore(IniFileStore):
    """IniFileStore that loads files from a transport.

    :ivar transport: The transport object where the config file is located.

    :ivar file_name: The config file basename in the transport directory.
    """

    def __init__(self, transport, file_name):
        """A Store using a ini file on a Transport

        Args:
          transport: The transport object where the config file is located.
          file_name: The config file basename in the transport directory.
        """
        super().__init__()
        self.transport = transport
        self.file_name = file_name

    def _load_content(self):
        try:
            return self.transport.get_bytes(self.file_name)
        except errors.PermissionDenied:
            trace.warning('Permission denied while trying to load configuration store %s.', self.external_url())
            raise

    def _save_content(self, content):
        self.transport.put_bytes(self.file_name, content)

    def external_url(self):
        return urlutils.join(self.transport.external_url(), urlutils.escape(self.file_name))