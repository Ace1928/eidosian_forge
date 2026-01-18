from contextlib import closing, contextmanager
import errno
import socket
import threading
import time
import http.client
import pytest
import cheroot.server
from cheroot.test import webtest
import cheroot.wsgi
class _TestClient:

    def __init__(self, server):
        self._interface, self._host, self._port = _get_conn_data(server.bind_addr)
        self.server_instance = server
        self._http_connection = self.get_connection()

    def get_connection(self):
        name = '{interface}:{port}'.format(interface=self._interface, port=self._port)
        conn_cls = http.client.HTTPConnection if self.server_instance.ssl_adapter is None else http.client.HTTPSConnection
        return conn_cls(name)

    def request(self, uri, method='GET', headers=None, http_conn=None, protocol='HTTP/1.1'):
        return webtest.openURL(uri, method=method, headers=headers, host=self._host, port=self._port, http_conn=http_conn or self._http_connection, protocol=protocol)

    def __getattr__(self, attr_name):

        def _wrapper(uri, **kwargs):
            http_method = attr_name.upper()
            return self.request(uri, method=http_method, **kwargs)
        return _wrapper