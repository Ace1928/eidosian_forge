import logging
import os
import select
import socket
import subprocess
import sys
from contextlib import closing
from io import BufferedReader, BytesIO
from typing import (
from urllib.parse import quote as urlquote
from urllib.parse import unquote as urlunquote
from urllib.parse import urljoin, urlparse, urlunparse, urlunsplit
import dulwich
from .config import Config, apply_instead_of, get_xdg_config_home_path
from .errors import GitProtocolError, NotGitRepository, SendPackError
from .pack import (
from .protocol import (
from .refs import PEELED_TAG_SUFFIX, _import_remote_refs, read_info_refs
from .repo import Repo
class TCPGitClient(TraditionalGitClient):
    """A Git Client that works over TCP directly (i.e. git://)."""

    def __init__(self, host, port=None, **kwargs) -> None:
        if port is None:
            port = TCP_GIT_PORT
        self._host = host
        self._port = port
        super().__init__(**kwargs)

    @classmethod
    def from_parsedurl(cls, parsedurl, **kwargs):
        return cls(parsedurl.hostname, port=parsedurl.port, **kwargs)

    def get_url(self, path):
        netloc = self._host
        if self._port is not None and self._port != TCP_GIT_PORT:
            netloc += ':%d' % self._port
        return urlunsplit(('git', netloc, path, '', ''))

    def _connect(self, cmd, path):
        if not isinstance(cmd, bytes):
            raise TypeError(cmd)
        if not isinstance(path, bytes):
            path = path.encode(self._remote_path_encoding)
        sockaddrs = socket.getaddrinfo(self._host, self._port, socket.AF_UNSPEC, socket.SOCK_STREAM)
        s = None
        err = OSError('no address found for %s' % self._host)
        for family, socktype, proto, canonname, sockaddr in sockaddrs:
            s = socket.socket(family, socktype, proto)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                s.connect(sockaddr)
                break
            except OSError as e:
                err = e
                if s is not None:
                    s.close()
                s = None
        if s is None:
            raise err
        rfile = s.makefile('rb', -1)
        wfile = s.makefile('wb', 0)

        def close():
            rfile.close()
            wfile.close()
            s.close()
        proto = Protocol(rfile.read, wfile.write, close, report_activity=self._report_activity)
        if path.startswith(b'/~'):
            path = path[1:]
        proto.send_cmd(b'git-' + cmd, path, b'host=' + self._host.encode('ascii'))
        return (proto, lambda: _fileno_can_read(s), None)