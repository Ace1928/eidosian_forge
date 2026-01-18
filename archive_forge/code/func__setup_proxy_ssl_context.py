import logging
import os
import os.path
import socket
import sys
import warnings
from base64 import b64encode
from urllib3 import PoolManager, Timeout, proxy_from_url
from urllib3.exceptions import (
from urllib3.exceptions import (
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError
from urllib3.exceptions import SSLError as URLLib3SSLError
from urllib3.util.retry import Retry
from urllib3.util.ssl_ import (
from urllib3.util.url import parse_url
import botocore.awsrequest
from botocore.compat import (
from botocore.exceptions import (
def _setup_proxy_ssl_context(self, proxy_url):
    proxies_settings = self._proxy_config.settings
    proxy_ca_bundle = proxies_settings.get('proxy_ca_bundle')
    proxy_cert = proxies_settings.get('proxy_client_cert')
    if proxy_ca_bundle is None and proxy_cert is None:
        return None
    context = self._get_ssl_context()
    try:
        url = parse_url(proxy_url)
        if not _is_ipaddress(url.host):
            context.check_hostname = True
        if proxy_ca_bundle is not None:
            context.load_verify_locations(cafile=proxy_ca_bundle)
        if isinstance(proxy_cert, tuple):
            context.load_cert_chain(proxy_cert[0], keyfile=proxy_cert[1])
        elif isinstance(proxy_cert, str):
            context.load_cert_chain(proxy_cert)
        return context
    except (OSError, URLLib3SSLError, LocationParseError) as e:
        raise InvalidProxiesConfigError(error=e)