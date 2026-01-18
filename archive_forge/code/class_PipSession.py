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
class PipSession(requests.Session):
    timeout: Optional[int] = None

    def __init__(self, *args: Any, retries: int=0, cache: Optional[str]=None, trusted_hosts: Sequence[str]=(), index_urls: Optional[List[str]]=None, ssl_context: Optional['SSLContext']=None, **kwargs: Any) -> None:
        """
        :param trusted_hosts: Domains not to emit warnings for when not using
            HTTPS.
        """
        super().__init__(*args, **kwargs)
        self.pip_trusted_origins: List[Tuple[str, Optional[int]]] = []
        self.headers['User-Agent'] = user_agent()
        self.auth = MultiDomainBasicAuth(index_urls=index_urls)
        retries = urllib3.Retry(total=retries, status_forcelist=[500, 502, 503, 520, 527], backoff_factor=0.25)
        insecure_adapter = InsecureHTTPAdapter(max_retries=retries)
        if cache:
            secure_adapter = CacheControlAdapter(cache=SafeFileCache(cache), max_retries=retries, ssl_context=ssl_context)
            self._trusted_host_adapter = InsecureCacheControlAdapter(cache=SafeFileCache(cache), max_retries=retries)
        else:
            secure_adapter = HTTPAdapter(max_retries=retries, ssl_context=ssl_context)
            self._trusted_host_adapter = insecure_adapter
        self.mount('https://', secure_adapter)
        self.mount('http://', insecure_adapter)
        self.mount('file://', LocalFSAdapter())
        for host in trusted_hosts:
            self.add_trusted_host(host, suppress_logging=True)

    def update_index_urls(self, new_index_urls: List[str]) -> None:
        """
        :param new_index_urls: New index urls to update the authentication
            handler with.
        """
        self.auth.index_urls = new_index_urls

    def add_trusted_host(self, host: str, source: Optional[str]=None, suppress_logging: bool=False) -> None:
        """
        :param host: It is okay to provide a host that has previously been
            added.
        :param source: An optional source string, for logging where the host
            string came from.
        """
        if not suppress_logging:
            msg = f'adding trusted host: {host!r}'
            if source is not None:
                msg += f' (from {source})'
            logger.info(msg)
        parsed_host, parsed_port = parse_netloc(host)
        if parsed_host is None:
            raise ValueError(f'Trusted host URL must include a host part: {host!r}')
        if (parsed_host, parsed_port) not in self.pip_trusted_origins:
            self.pip_trusted_origins.append((parsed_host, parsed_port))
        self.mount(build_url_from_netloc(host, scheme='http') + '/', self._trusted_host_adapter)
        self.mount(build_url_from_netloc(host) + '/', self._trusted_host_adapter)
        if not parsed_port:
            self.mount(build_url_from_netloc(host, scheme='http') + ':', self._trusted_host_adapter)
            self.mount(build_url_from_netloc(host) + ':', self._trusted_host_adapter)

    def iter_secure_origins(self) -> Generator[SecureOrigin, None, None]:
        yield from SECURE_ORIGINS
        for host, port in self.pip_trusted_origins:
            yield ('*', host, '*' if port is None else port)

    def is_secure_origin(self, location: Link) -> bool:
        parsed = urllib.parse.urlparse(str(location))
        origin_protocol, origin_host, origin_port = (parsed.scheme, parsed.hostname, parsed.port)
        origin_protocol = origin_protocol.rsplit('+', 1)[-1]
        for secure_origin in self.iter_secure_origins():
            secure_protocol, secure_host, secure_port = secure_origin
            if origin_protocol != secure_protocol and secure_protocol != '*':
                continue
            try:
                addr = ipaddress.ip_address(origin_host or '')
                network = ipaddress.ip_network(secure_host)
            except ValueError:
                if origin_host and origin_host.lower() != secure_host.lower() and (secure_host != '*'):
                    continue
            else:
                if addr not in network:
                    continue
            if origin_port != secure_port and secure_port != '*' and (secure_port is not None):
                continue
            return True
        logger.warning("The repository located at %s is not a trusted or secure host and is being ignored. If this repository is available via HTTPS we recommend you use HTTPS instead, otherwise you may silence this warning and allow it anyway with '--trusted-host %s'.", origin_host, origin_host)
        return False

    def request(self, method: str, url: str, *args: Any, **kwargs: Any) -> Response:
        kwargs.setdefault('timeout', self.timeout)
        kwargs.setdefault('proxies', self.proxies)
        return super().request(method, url, *args, **kwargs)