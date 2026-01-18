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
def _win32_url_to_path(parsed) -> str:
    """Convert a file: URL to a path.

    https://datatracker.ietf.org/doc/html/rfc8089
    """
    assert sys.platform == 'win32' or os.name == 'nt'
    assert parsed.scheme == 'file'
    _, netloc, path, _, _, _ = parsed
    if netloc == 'localhost' or not netloc:
        netloc = ''
    elif netloc and len(netloc) >= 2 and netloc[0].isalpha() and (netloc[1:2] in (':', ':/')):
        netloc = netloc[:2]
    else:
        raise NotImplementedError('Non-local file URLs are not supported')
    global url2pathname
    if url2pathname is None:
        from urllib.request import url2pathname
    return url2pathname(netloc + path)