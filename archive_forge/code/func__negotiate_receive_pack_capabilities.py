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
def _negotiate_receive_pack_capabilities(self, server_capabilities):
    negotiated_capabilities = self._send_capabilities & server_capabilities
    agent = None
    for capability in server_capabilities:
        k, v = parse_capability(capability)
        if k == CAPABILITY_AGENT:
            agent = v
    extract_capability_names(server_capabilities) - KNOWN_RECEIVE_CAPABILITIES
    return (negotiated_capabilities, agent)