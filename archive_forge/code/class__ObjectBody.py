import socket
import re
import logging
import warnings
from requests.exceptions import RequestException, SSLError
import http.client as http_client
from urllib.parse import quote, unquote
from urllib.parse import urljoin, urlparse, urlunparse
from time import sleep, time
from swiftclient import version as swiftclient_version
from swiftclient.exceptions import ClientException
from swiftclient.requests_compat import SwiftClientRequestsSession
from swiftclient.utils import (
class _ObjectBody:
    """
    Readable and iterable object body response wrapper.
    """

    def __init__(self, resp, chunk_size, conn_to_close):
        """
        Wrap the underlying response

        :param resp: the response to wrap
        :param chunk_size: number of bytes to return each iteration/next call
        """
        self.resp = resp
        self.chunk_size = chunk_size
        self.conn_to_close = conn_to_close

    def read(self, length=None):
        buf = self.resp.read(length)
        if length != 0 and (not buf):
            self.close()
        return buf

    def __iter__(self):
        return self

    def next(self):
        buf = self.read(self.chunk_size)
        if not buf:
            raise StopIteration()
        return buf

    def __next__(self):
        return self.next()

    def close(self):
        self.resp.close()
        if self.conn_to_close:
            self.conn_to_close.close()