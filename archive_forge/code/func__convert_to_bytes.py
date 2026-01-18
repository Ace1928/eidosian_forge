import functools
import logging
from collections.abc import Mapping
import urllib3.util
from urllib3.connection import HTTPConnection, VerifiedHTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
import botocore.utils
from botocore.compat import (
from botocore.exceptions import UnseekableStreamError
def _convert_to_bytes(self, mixed_buffer):
    bytes_buffer = []
    for chunk in mixed_buffer:
        if isinstance(chunk, str):
            bytes_buffer.append(chunk.encode('utf-8'))
        else:
            bytes_buffer.append(chunk)
    msg = b'\r\n'.join(bytes_buffer)
    return msg