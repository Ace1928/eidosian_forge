import errno
import os
import socket
import sys
import six
from ._exceptions import *
from ._logging import *
from ._socket import*
from ._ssl_compat import *
from ._url import *
class proxy_info(object):

    def __init__(self, **options):
        self.type = options.get('proxy_type') or 'http'
        if not self.type in ['http', 'socks4', 'socks5', 'socks5h']:
            raise ValueError("proxy_type must be 'http', 'socks4', 'socks5' or 'socks5h'")
        self.host = options.get('http_proxy_host', None)
        if self.host:
            self.port = options.get('http_proxy_port', 0)
            self.auth = options.get('http_proxy_auth', None)
            self.no_proxy = options.get('http_no_proxy', None)
        else:
            self.port = 0
            self.auth = None
            self.no_proxy = None