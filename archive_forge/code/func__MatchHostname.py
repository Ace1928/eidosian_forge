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
def _MatchHostname(self, cert, hostname):
    with self._lock:
        assert self._ip_to_host is not None
        if hostname in self._ip_to_host:
            hostname = self._ip_to_host[hostname]
    return self._old_match_hostname(cert, hostname)