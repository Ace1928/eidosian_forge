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
def _ssl_socket(sock, user_sslopt, hostname):
    sslopt = dict(cert_reqs=ssl.CERT_REQUIRED)
    sslopt.update(user_sslopt)
    certPath = os.environ.get('WEBSOCKET_CLIENT_CA_BUNDLE')
    if certPath and os.path.isfile(certPath) and (user_sslopt.get('ca_certs', None) is None) and (user_sslopt.get('ca_cert', None) is None):
        sslopt['ca_certs'] = certPath
    elif certPath and os.path.isdir(certPath) and (user_sslopt.get('ca_cert_path', None) is None):
        sslopt['ca_cert_path'] = certPath
    check_hostname = sslopt['cert_reqs'] != ssl.CERT_NONE and sslopt.pop('check_hostname', True)
    if _can_use_sni():
        sock = _wrap_sni_socket(sock, sslopt, hostname, check_hostname)
    else:
        sslopt.pop('check_hostname', True)
        sock = ssl.wrap_socket(sock, **sslopt)
    if not HAVE_CONTEXT_CHECK_HOSTNAME and check_hostname:
        match_hostname(sock.getpeercert(), hostname)
    return sock