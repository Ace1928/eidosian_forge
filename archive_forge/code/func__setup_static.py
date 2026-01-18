from __future__ import annotations
import json
import configparser
import os
import urllib.parse
import typing as t
from ....util import (
from ....config import (
from ....docker_util import (
from ....containers import (
from . import (
def _setup_static(self) -> None:
    """Configure CloudStack tests for use with static configuration."""
    parser = configparser.ConfigParser()
    parser.read(self.config_static_path)
    endpoint = parser.get('cloudstack', 'endpoint')
    parts = urllib.parse.urlparse(endpoint)
    self.host = parts.hostname
    if not self.host:
        raise ApplicationError('Could not determine host from endpoint: %s' % endpoint)
    if parts.port:
        self.port = parts.port
    elif parts.scheme == 'http':
        self.port = 80
    elif parts.scheme == 'https':
        self.port = 443
    else:
        raise ApplicationError('Could not determine port from endpoint: %s' % endpoint)
    display.info('Read cs host "%s" and port %d from config: %s' % (self.host, self.port, self.config_static_path), verbosity=1)