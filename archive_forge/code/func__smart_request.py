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
def _smart_request(self, service, url, data):
    """Send a 'smart' HTTP request.

        This is a simple wrapper around _http_request that sets
        a couple of extra headers.
        """
    assert url[-1] == '/'
    url = urljoin(url, service)
    result_content_type = 'application/x-%s-result' % service
    headers = {'Content-Type': 'application/x-%s-request' % service, 'Accept': result_content_type}
    if isinstance(data, bytes):
        headers['Content-Length'] = str(len(data))
    resp, read = self._http_request(url, headers, data)
    if resp.content_type.split(';')[0] != result_content_type:
        raise GitProtocolError('Invalid content-type from server: %s' % resp.content_type)
    return (resp, read)