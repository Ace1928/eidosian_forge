import asyncio
import http.client
import io
import json
import math
import re
import sys
import tempfile
import xmlrpc.client
from datetime import datetime, timedelta, timezone
from functools import partial
from itertools import groupby
from os import environ
from pathlib import Path
from subprocess import CalledProcessError, run
from tarfile import TarFile
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from zipfile import ZipFile
import httpx
import tornado
from async_lru import alru_cache
from traitlets import CFloat, CInt, Unicode, config, observe
from jupyterlab._version import __version__
from jupyterlab.extensions.manager import (
class ProxiedTransport(xmlrpc.client.Transport):

    def set_proxy(self, host, port=None, headers=None):
        self.proxy = (host, port)
        self.proxy_headers = headers

    def make_connection(self, host):
        connection = http.client.HTTPConnection(*self.proxy)
        connection.set_tunnel(host, headers=self.proxy_headers)
        self._connection = (host, connection)
        return connection