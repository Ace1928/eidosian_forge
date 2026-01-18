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
def _remote_error_from_stderr(stderr):
    if stderr is None:
        return HangupException()
    lines = [line.rstrip(b'\n') for line in stderr.readlines()]
    for line in lines:
        if line.startswith(b'ERROR: '):
            return GitProtocolError(line[len(b'ERROR: '):].decode('utf-8', 'replace'))
    return HangupException(lines)