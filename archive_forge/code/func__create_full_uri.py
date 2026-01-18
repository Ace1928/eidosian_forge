from collections import abc
import errno
import socket
import ssl
import warnings
import httplib2
import six.moves.http_client
import urllib3
@classmethod
def _create_full_uri(cls, conn, request_uri):
    if isinstance(conn, six.moves.http_client.HTTPSConnection):
        scheme = 'https'
    else:
        scheme = 'http'
    host = conn.host
    if _is_ipv6(host):
        host = '[{}]'.format(host)
    port = ''
    if conn.port is not None:
        port = ':{}'.format(conn.port)
    return '{}://{}{}{}'.format(scheme, host, port, request_uri)