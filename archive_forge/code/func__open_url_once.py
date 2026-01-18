from your HTTP server.
import pprint
import re
import socket
import sys
import time
import traceback
import os
import json
import unittest  # pylint: disable=deprecated-module,preferred-module
import warnings
import functools
import http.client
import urllib.parse
from more_itertools.more import always_iterable
import jaraco.functools
def _open_url_once(url, headers=None, method='GET', body=None, host='127.0.0.1', port=8000, http_conn=http.client.HTTPConnection, protocol='HTTP/1.1', ssl_context=None):
    """Open the given HTTP resource and return status, headers, and body."""
    headers = cleanHeaders(headers, method, body, host, port)
    if hasattr(http_conn, 'host'):
        conn = http_conn
    else:
        kw = {}
        if ssl_context:
            kw['context'] = ssl_context
        conn = http_conn(interface(host), port, **kw)
    conn._http_vsn_str = protocol
    conn._http_vsn = int(''.join([x for x in protocol if x.isdigit()]))
    if isinstance(url, bytes):
        url = url.decode()
    conn.putrequest(method.upper(), url, skip_host=True, skip_accept_encoding=True)
    for key, value in headers:
        conn.putheader(key, value.encode('Latin-1'))
    conn.endheaders()
    if body is not None:
        conn.send(body)
    response = conn.getresponse()
    s, h, b = shb(response)
    if not hasattr(http_conn, 'host'):
        conn.close()
    return (s, h, b)