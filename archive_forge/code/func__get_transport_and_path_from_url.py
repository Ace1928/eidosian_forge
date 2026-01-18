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
def _get_transport_and_path_from_url(url, config, operation, **kwargs):
    parsed = urlparse(url)
    if parsed.scheme == 'git':
        return (TCPGitClient.from_parsedurl(parsed, **kwargs), parsed.path)
    elif parsed.scheme in ('git+ssh', 'ssh'):
        return (SSHGitClient.from_parsedurl(parsed, **kwargs), parsed.path)
    elif parsed.scheme in ('http', 'https'):
        return (HttpGitClient.from_parsedurl(parsed, config=config, **kwargs), parsed.path)
    elif parsed.scheme == 'file':
        if sys.platform == 'win32' or os.name == 'nt':
            return (default_local_git_client_cls(**kwargs), _win32_url_to_path(parsed))
        return (default_local_git_client_cls.from_parsedurl(parsed, **kwargs), parsed.path)
    raise ValueError("unknown scheme '%s'" % parsed.scheme)