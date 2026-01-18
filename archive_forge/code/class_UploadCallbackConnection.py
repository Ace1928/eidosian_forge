from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import copy
import logging
import re
import socket
import types
import six
from six.moves import http_client
from six.moves import urllib
from six.moves import cStringIO
from apitools.base.py import exceptions as apitools_exceptions
from gslib.cloud_api import BadRequestException
from gslib.lazy_wrapper import LazyWrapper
from gslib.progress_callback import ProgressCallbackWithTimeout
from gslib.utils.constants import DEBUGLEVEL_DUMP_REQUESTS
from gslib.utils.constants import SSL_TIMEOUT_SEC
from gslib.utils.constants import TRANSFER_BUFFER_SIZE
from gslib.utils.constants import UTF8
from gslib.utils import text_util
import httplib2
from httplib2 import parse_uri
class UploadCallbackConnection(httplib2.HTTPSConnectionWithTimeout):
    """Connection class override for uploads."""
    bytes_uploaded_container = outer_bytes_uploaded_container
    processed_initial_bytes = False
    GCS_JSON_BUFFER_SIZE = outer_buffer_size
    callback_processor = None
    size = outer_total_size
    header_encoding = ''
    header_length = None
    header_range = None
    size_modifier = 1.0

    def __init__(self, *args, **kwargs):
        kwargs['timeout'] = SSL_TIMEOUT_SEC
        httplib2.HTTPSConnectionWithTimeout.__init__(self, *args, **kwargs)

    def _send_output(self, message_body=None, encode_chunked=False):
        """Send the currently buffered request and clear the buffer.

        Appends an extra \\r\\n to the buffer.

        Args:
          message_body: if specified, this is appended to the request.
        """
        self._buffer.extend((b'', b''))
        if six.PY2:
            items = self._buffer
        else:
            items = []
            for item in self._buffer:
                if isinstance(item, bytes):
                    items.append(item)
                else:
                    items.append(item.encode(UTF8))
        msg = b'\r\n'.join(items)
        num_metadata_bytes = len(msg)
        if outer_debug == DEBUGLEVEL_DUMP_REQUESTS and outer_logger:
            outer_logger.debug('send: %s' % msg)
        del self._buffer[:]
        if isinstance(message_body, str):
            msg += message_body
            message_body = None
        self.send(msg, num_metadata_bytes=num_metadata_bytes)
        if message_body is not None:
            self.send(message_body)

    def putheader(self, header, *values):
        """Overrides HTTPConnection.putheader.

        Send a request header line to the server. For example:
        h.putheader('Accept', 'text/html').

        This override records the content encoding, length, and range of the
        payload. For uploads where the content-range difference does not match
        the content-length, progress printing will under-report progress. These
        headers are used to calculate a multiplier to correct the progress.

        For example: the content-length for gzip transport encoded data
        represents the compressed size of the data while the content-range
        difference represents the uncompressed size. Dividing the
        content-range difference by the content-length gives the ratio to
        multiply the progress by to correctly report the relative progress.

        Args:
          header: The header.
          *values: A set of values for the header.
        """
        if header == 'content-encoding':
            value = ''.join([str(v) for v in values])
            self.header_encoding = value
            if outer_debug == DEBUGLEVEL_DUMP_REQUESTS and outer_logger:
                outer_logger.debug('send: Using gzip transport encoding for the request.')
        elif header == 'content-length':
            try:
                value = int(''.join([str(v) for v in values]))
                self.header_length = value
            except ValueError:
                pass
        elif header == 'content-range':
            try:
                value = ''.join([str(v) for v in values])
                ranges = DECIMAL_REGEX().findall(value)
                if len(ranges) > 1:
                    self.header_range = int(ranges[1]) - int(ranges[0]) + 1
            except ValueError:
                pass
        if self.header_encoding == 'gzip' and self.header_length and self.header_range:
            self.size_modifier = self.header_range / float(self.header_length)
            self.header_encoding = ''
            self.header_length = None
            self.header_range = None
            if outer_debug == DEBUGLEVEL_DUMP_REQUESTS and outer_logger:
                outer_logger.debug('send: Setting progress modifier to %s.' % self.size_modifier)
        http_client.HTTPSConnection.putheader(self, header, *values)

    def send(self, data, num_metadata_bytes=0):
        """Overrides HTTPConnection.send.

        Args:
          data: string or file-like object (implements read()) of data to send.
          num_metadata_bytes: number of bytes that consist of metadata
              (headers, etc.) not representing the data being uploaded.
        """
        if not self.processed_initial_bytes:
            self.processed_initial_bytes = True
            if outer_progress_callback:
                self.callback_processor = ProgressCallbackWithTimeout(outer_total_size, outer_progress_callback)
                self.callback_processor.Progress(self.bytes_uploaded_container.bytes_transferred)
        if isinstance(data, six.text_type):
            full_buffer = cStringIO(data)
        elif isinstance(data, six.binary_type):
            full_buffer = six.BytesIO(data)
        else:
            full_buffer = data
        partial_buffer = full_buffer.read(self.GCS_JSON_BUFFER_SIZE)
        while partial_buffer:
            if six.PY2:
                httplib2.HTTPSConnectionWithTimeout.send(self, partial_buffer)
            elif isinstance(partial_buffer, bytes):
                httplib2.HTTPSConnectionWithTimeout.send(self, partial_buffer)
            else:
                httplib2.HTTPSConnectionWithTimeout.send(self, partial_buffer.encode(UTF8))
            sent_data_bytes = len(partial_buffer)
            if num_metadata_bytes:
                if num_metadata_bytes <= sent_data_bytes:
                    sent_data_bytes -= num_metadata_bytes
                    num_metadata_bytes = 0
                else:
                    num_metadata_bytes -= sent_data_bytes
                    sent_data_bytes = 0
            if self.callback_processor:
                sent_data_bytes = int(sent_data_bytes * self.size_modifier)
                self.callback_processor.Progress(sent_data_bytes)
            partial_buffer = full_buffer.read(self.GCS_JSON_BUFFER_SIZE)