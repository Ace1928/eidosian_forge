from __future__ import print_function
import email.generator as email_generator
import email.mime.multipart as mime_multipart
import email.mime.nonmultipart as mime_nonmultipart
import io
import json
import mimetypes
import os
import threading
import six
from six.moves import http_client
from apitools.base.py import buffered_stream
from apitools.base.py import compression
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import stream_slice
from apitools.base.py import util
def __GetChunk(self, start, end, additional_headers=None):
    """Retrieve a chunk, and return the full response."""
    self.EnsureInitialized()
    request = http_wrapper.Request(url=self.url)
    self.__SetRangeHeader(request, start, end=end)
    if additional_headers is not None:
        request.headers.update(additional_headers)
    return http_wrapper.MakeRequest(self.bytes_http, request, retry_func=self.retry_func, retries=self.num_retries)