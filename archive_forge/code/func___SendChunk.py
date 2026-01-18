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
def __SendChunk(self, start, additional_headers=None):
    """Send the specified chunk."""
    self.EnsureInitialized()
    no_log_body = self.total_size is None
    request = http_wrapper.Request(url=self.url, http_method='PUT')
    if self.__gzip_encoded:
        request.headers['Content-Encoding'] = 'gzip'
        body_stream, read_length, exhausted = compression.CompressStream(self.stream, self.chunksize)
        end = start + read_length
        if self.total_size is None and exhausted:
            self.__total_size = end
    elif self.total_size is None:
        body_stream = buffered_stream.BufferedStream(self.stream, start, self.chunksize)
        end = body_stream.stream_end_position
        if body_stream.stream_exhausted:
            self.__total_size = end
        body_stream = body_stream.read(self.chunksize)
    else:
        end = min(start + self.chunksize, self.total_size)
        body_stream = stream_slice.StreamSlice(self.stream, end - start)
    request.body = body_stream
    request.headers['Content-Type'] = self.mime_type
    if no_log_body:
        request.loggable_body = '<media body>'
    if self.total_size is None:
        range_string = 'bytes %s-%s/*' % (start, end - 1)
    elif end == start:
        range_string = 'bytes */%s' % self.total_size
    else:
        range_string = 'bytes %s-%s/%s' % (start, end - 1, self.total_size)
    request.headers['Content-Range'] = range_string
    if additional_headers:
        request.headers.update(additional_headers)
    return self.__SendMediaRequest(request, end)