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
def GetRange(self, start, end=None, additional_headers=None, use_chunks=True):
    """Retrieve a given byte range from this download, inclusive.

        Range must be of one of these three forms:
        * 0 <= start, end = None: Fetch from start to the end of the file.
        * 0 <= start <= end: Fetch the bytes from start to end.
        * start < 0, end = None: Fetch the last -start bytes of the file.

        (These variations correspond to those described in the HTTP 1.1
        protocol for range headers in RFC 2616, sec. 14.35.1.)

        Args:
          start: (int) Where to start fetching bytes. (See above.)
          end: (int, optional) Where to stop fetching bytes. (See above.)
          additional_headers: (bool, optional) Any additional headers to
              pass with the request.
          use_chunks: (bool, default: True) If False, ignore self.chunksize
              and fetch this range in a single request.

        Returns:
          None. Streams bytes into self.stream.
        """
    self.EnsureInitialized()
    progress_end_normalized = False
    if self.total_size is not None:
        progress, end_byte = self.__NormalizeStartEnd(start, end)
        progress_end_normalized = True
    else:
        progress = start
        end_byte = end
    while not progress_end_normalized or end_byte is None or progress <= end_byte:
        end_byte = self.__ComputeEndByte(progress, end=end_byte, use_chunks=use_chunks)
        response = self.__GetChunk(progress, end_byte, additional_headers=additional_headers)
        if not progress_end_normalized:
            self.__SetTotal(response.info)
            progress, end_byte = self.__NormalizeStartEnd(start, end)
            progress_end_normalized = True
        response = self.__ProcessResponse(response)
        progress += response.length
        if response.length == 0:
            if response.status_code == http_client.OK:
                return
            raise exceptions.TransferRetryError('Zero bytes unexpectedly returned in download response')