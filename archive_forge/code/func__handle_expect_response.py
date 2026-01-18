import functools
import logging
from collections.abc import Mapping
import urllib3.util
from urllib3.connection import HTTPConnection, VerifiedHTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
import botocore.utils
from botocore.compat import (
from botocore.exceptions import UnseekableStreamError
def _handle_expect_response(self, message_body):
    fp = self.sock.makefile('rb', 0)
    try:
        maybe_status_line = fp.readline()
        parts = maybe_status_line.split(None, 2)
        if self._is_100_continue_status(maybe_status_line):
            self._consume_headers(fp)
            logger.debug('100 Continue response seen, now sending request body.')
            self._send_message_body(message_body)
        elif len(parts) == 3 and parts[0].startswith(b'HTTP/'):
            logger.debug('Received a non 100 Continue response from the server, NOT sending request body.')
            status_tuple = (parts[0].decode('ascii'), int(parts[1]), parts[2].decode('ascii'))
            response_class = functools.partial(AWSHTTPResponse, status_tuple=status_tuple)
            self.response_class = response_class
            self._response_received = True
    finally:
        fp.close()