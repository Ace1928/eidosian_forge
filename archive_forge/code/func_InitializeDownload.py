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
def InitializeDownload(self, http_request, http=None, client=None):
    """Initialize this download by making a request.

        Args:
          http_request: The HttpRequest to use to initialize this download.
          http: The httplib2.Http instance for this request.
          client: If provided, let this client process the final URL before
              sending any additional requests. If client is provided and
              http is not, client.http will be used instead.
        """
    self.EnsureUninitialized()
    if http is None and client is None:
        raise exceptions.UserError('Must provide client or http.')
    http = http or client.http
    if client is not None:
        http_request.url = client.FinalizeTransferUrl(http_request.url)
    url = http_request.url
    if self.auto_transfer:
        end_byte = self.__ComputeEndByte(0)
        self.__SetRangeHeader(http_request, 0, end_byte)
        response = http_wrapper.MakeRequest(self.bytes_http or http, http_request)
        if response.status_code not in self._ACCEPTABLE_STATUSES:
            raise exceptions.HttpError.FromResponse(response)
        self.__initial_response = response
        self.__SetTotal(response.info)
        url = response.info.get('content-location', response.request_url)
    if client is not None:
        url = client.FinalizeTransferUrl(url)
    self._Initialize(http, url)
    if self.auto_transfer:
        self.StreamInChunks()