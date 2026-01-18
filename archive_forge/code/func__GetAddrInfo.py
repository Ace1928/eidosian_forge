from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import base64
import contextlib
import os
import socket
import ssl
import tempfile
import threading
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import files
def _GetAddrInfo(self, host, *args, **kwargs):
    """Like socket.getaddrinfo, only with translation."""
    with self._lock:
        assert self._host_to_ip is not None
        if host in self._host_to_ip:
            host = self._host_to_ip[host]
    return self._old_getaddrinfo(host, *args, **kwargs)