import os
import socket
import struct
from six.moves.urllib.parse import urlparse

    try to retrieve proxy host and port from environment
    if not provided in options.
    result is (proxy_host, proxy_port, proxy_auth).
    proxy_auth is tuple of username and password
     of proxy authentication information.

    hostname: websocket server name.

    is_secure:  is the connection secure? (wss)
                looks for "https_proxy" in env
                before falling back to "http_proxy"

    options:    "http_proxy_host" - http proxy host name.
                "http_proxy_port" - http proxy port.
                "http_no_proxy"   - host names, which doesn't use proxy.
                "http_proxy_auth" - http proxy auth information.
                                    tuple of username and password.
                                    default is None
                "proxy_type"      - if set to "socks5" PySocks wrapper
                                    will be used in place of a http proxy.
                                    default is "http"
    