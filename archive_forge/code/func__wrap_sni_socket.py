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
def _wrap_sni_socket(sock, sslopt, hostname, check_hostname):
    context = ssl.SSLContext(sslopt.get('ssl_version', ssl.PROTOCOL_SSLv23))
    if sslopt.get('cert_reqs', ssl.CERT_NONE) != ssl.CERT_NONE:
        cafile = sslopt.get('ca_certs', None)
        capath = sslopt.get('ca_cert_path', None)
        if cafile or capath:
            context.load_verify_locations(cafile=cafile, capath=capath)
        elif hasattr(context, 'load_default_certs'):
            context.load_default_certs(ssl.Purpose.SERVER_AUTH)
    if sslopt.get('certfile', None):
        context.load_cert_chain(sslopt['certfile'], sslopt.get('keyfile', None), sslopt.get('password', None))
    context.verify_mode = sslopt['cert_reqs']
    if HAVE_CONTEXT_CHECK_HOSTNAME:
        context.check_hostname = check_hostname
    if 'ciphers' in sslopt:
        context.set_ciphers(sslopt['ciphers'])
    if 'cert_chain' in sslopt:
        certfile, keyfile, password = sslopt['cert_chain']
        context.load_cert_chain(certfile, keyfile, password)
    if 'ecdh_curve' in sslopt:
        context.set_ecdh_curve(sslopt['ecdh_curve'])
    return context.wrap_socket(sock, do_handshake_on_connect=sslopt.get('do_handshake_on_connect', True), suppress_ragged_eofs=sslopt.get('suppress_ragged_eofs', True), server_hostname=hostname)