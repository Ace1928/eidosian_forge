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
@contextlib.contextmanager
def MonkeypatchAddressChecking(self, hostname, ip):
    """Change ssl address checking so the given ip answers to the hostname."""
    with self._lock:
        match_hostname_exists = hasattr(ssl, 'match_hostname')
        if self._host_to_ip is None:
            self._host_to_ip = {}
            self._ip_to_host = {}
            if match_hostname_exists:
                self._old_match_hostname = ssl.match_hostname
                ssl.match_hostname = self._MatchHostname
            self._old_getaddrinfo = socket.getaddrinfo
        if hostname in self._host_to_ip:
            raise ValueError('Cannot re-patch the same address: {}'.format(hostname))
        if ip in self._ip_to_host:
            raise ValueError('Cannot re-patch the same address: {}'.format(ip))
        self._host_to_ip[hostname] = ip
        self._ip_to_host[ip] = hostname
    try:
        yield ip
    finally:
        with self._lock:
            del self._host_to_ip[hostname]
            del self._ip_to_host[ip]
            if not self._host_to_ip:
                self._host_to_ip = None
                self._ip_to_host = None
                if match_hostname_exists:
                    ssl.match_hostname = self._old_match_hostname