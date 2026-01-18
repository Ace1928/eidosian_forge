import errno
import gc
import logging
import os
import pprint
import sys
import tempfile
import traceback
import eventlet.backdoor
import greenlet
import yappi
from eventlet.green import socket
from oslo_service._i18n import _
from oslo_service import _options
def _try_open_unix_domain_socket(socket_path):
    try:
        return eventlet.listen(socket_path, socket.AF_UNIX)
    except socket.error as e:
        if e.errno != errno.EADDRINUSE:
            raise
        else:
            try:
                os.unlink(socket_path)
            except OSError as e:
                if e.errno != errno.ENOENT:
                    raise
            return eventlet.listen(socket_path, socket.AF_UNIX)