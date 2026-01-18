import errno
import os
import socket
import struct
import sys
import traceback
import warnings
from . import _auth
from .charset import charset_by_name, charset_by_id
from .constants import CLIENT, COMMAND, CR, ER, FIELD_TYPE, SERVER_STATUS
from . import converters
from .cursors import Cursor
from .optionfile import Parser
from .protocol import (
from . import err, VERSION_STRING
def _get_auth_plugin_handler(self, plugin_name):
    plugin_class = self._auth_plugin_map.get(plugin_name)
    if not plugin_class and isinstance(plugin_name, bytes):
        plugin_class = self._auth_plugin_map.get(plugin_name.decode('ascii'))
    if plugin_class:
        try:
            handler = plugin_class(self)
        except TypeError:
            raise err.OperationalError(CR.CR_AUTH_PLUGIN_CANNOT_LOAD, "Authentication plugin '%s' not loaded: - %r cannot be constructed with connection object" % (plugin_name, plugin_class))
    else:
        handler = None
    return handler