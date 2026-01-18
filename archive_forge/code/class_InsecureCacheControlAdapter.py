import email.utils
import io
import ipaddress
import json
import logging
import mimetypes
import os
import platform
import shutil
import subprocess
import sys
import urllib.parse
import warnings
from typing import (
from pip._vendor import requests, urllib3
from pip._vendor.cachecontrol import CacheControlAdapter as _BaseCacheControlAdapter
from pip._vendor.requests.adapters import DEFAULT_POOLBLOCK, BaseAdapter
from pip._vendor.requests.adapters import HTTPAdapter as _BaseHTTPAdapter
from pip._vendor.requests.models import PreparedRequest, Response
from pip._vendor.requests.structures import CaseInsensitiveDict
from pip._vendor.urllib3.connectionpool import ConnectionPool
from pip._vendor.urllib3.exceptions import InsecureRequestWarning
from pip import __version__
from pip._internal.metadata import get_default_environment
from pip._internal.models.link import Link
from pip._internal.network.auth import MultiDomainBasicAuth
from pip._internal.network.cache import SafeFileCache
from pip._internal.utils.compat import has_tls
from pip._internal.utils.glibc import libc_ver
from pip._internal.utils.misc import build_url_from_netloc, parse_netloc
from pip._internal.utils.urls import url_to_path
class InsecureCacheControlAdapter(CacheControlAdapter):

    def cert_verify(self, conn: ConnectionPool, url: str, verify: Union[bool, str], cert: Optional[Union[str, Tuple[str, str]]]) -> None:
        super().cert_verify(conn=conn, url=url, verify=False, cert=cert)