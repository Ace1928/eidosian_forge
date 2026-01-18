from __future__ import absolute_import
from socket import error as SocketError
from socket import timeout as SocketTimeout
from ..connection import HTTPConnection, HTTPSConnection
from ..connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from ..exceptions import ConnectTimeoutError, NewConnectionError
from ..poolmanager import PoolManager
from ..util.url import parse_url
class SOCKSProxyManager(PoolManager):
    """
    A version of the urllib3 ProxyManager that routes connections via the
    defined SOCKS proxy.
    """
    pool_classes_by_scheme = {'http': SOCKSHTTPConnectionPool, 'https': SOCKSHTTPSConnectionPool}

    def __init__(self, proxy_url, username=None, password=None, num_pools=10, headers=None, **connection_pool_kw):
        parsed = parse_url(proxy_url)
        if username is None and password is None and (parsed.auth is not None):
            split = parsed.auth.split(':')
            if len(split) == 2:
                username, password = split
        if parsed.scheme == 'socks5':
            socks_version = socks.PROXY_TYPE_SOCKS5
            rdns = False
        elif parsed.scheme == 'socks5h':
            socks_version = socks.PROXY_TYPE_SOCKS5
            rdns = True
        elif parsed.scheme == 'socks4':
            socks_version = socks.PROXY_TYPE_SOCKS4
            rdns = False
        elif parsed.scheme == 'socks4a':
            socks_version = socks.PROXY_TYPE_SOCKS4
            rdns = True
        else:
            raise ValueError('Unable to determine SOCKS version from %s' % proxy_url)
        self.proxy_url = proxy_url
        socks_options = {'socks_version': socks_version, 'proxy_host': parsed.host, 'proxy_port': parsed.port, 'username': username, 'password': password, 'rdns': rdns}
        connection_pool_kw['_socks_options'] = socks_options
        super(SOCKSProxyManager, self).__init__(num_pools, headers, **connection_pool_kw)
        self.pool_classes_by_scheme = SOCKSProxyManager.pool_classes_by_scheme