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
class SubprocessWrapper:
    """A socket-like object that talks to a subprocess via pipes."""

    def __init__(self, proc) -> None:
        self.proc = proc
        self.read = BufferedReader(proc.stdout).read
        self.write = proc.stdin.write

    @property
    def stderr(self):
        return self.proc.stderr

    def can_read(self):
        if sys.platform == 'win32':
            from msvcrt import get_osfhandle
            handle = get_osfhandle(self.proc.stdout.fileno())
            return _win32_peek_avail(handle) != 0
        else:
            return _fileno_can_read(self.proc.stdout.fileno())

    def close(self):
        self.proc.stdin.close()
        self.proc.stdout.close()
        if self.proc.stderr:
            self.proc.stderr.close()
        self.proc.wait()