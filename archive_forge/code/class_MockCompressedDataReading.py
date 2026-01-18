import contextlib
import re
import socket
import ssl
import zlib
from base64 import b64decode
from io import BufferedReader, BytesIO, TextIOWrapper
from test import onlyBrotlipy
import mock
import pytest
import six
from urllib3.exceptions import (
from urllib3.packages.six.moves import http_client as httplib
from urllib3.response import HTTPResponse, brotli
from urllib3.util.response import is_fp_closed
from urllib3.util.retry import RequestHistory, Retry
class MockCompressedDataReading(BytesIO):
    """
            A BytesIO-like reader returning ``payload`` in ``NUMBER_OF_READS``
            calls to ``read``.
            """

    def __init__(self, payload, payload_part_size):
        self.payloads = [payload[i * payload_part_size:(i + 1) * payload_part_size] for i in range(NUMBER_OF_READS + 1)]
        assert b''.join(self.payloads) == payload

    def read(self, _):
        if len(self.payloads) > 0:
            return self.payloads.pop(0)
        return b''