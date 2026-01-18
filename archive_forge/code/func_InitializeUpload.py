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
def InitializeUpload(self, http_request, http=None, client=None):
    """Initialize this upload from the given http_request."""
    if self.strategy is None:
        raise exceptions.UserError('No upload strategy set; did you call ConfigureRequest?')
    if http is None and client is None:
        raise exceptions.UserError('Must provide client or http.')
    if self.strategy != RESUMABLE_UPLOAD:
        return
    http = http or client.http
    if client is not None:
        http_request.url = client.FinalizeTransferUrl(http_request.url)
    self.EnsureUninitialized()
    http_response = http_wrapper.MakeRequest(http, http_request, retries=self.num_retries)
    if http_response.status_code != http_client.OK:
        raise exceptions.HttpError.FromResponse(http_response)
    self.__server_chunk_granularity = http_response.info.get('X-Goog-Upload-Chunk-Granularity')
    url = http_response.info['location']
    if client is not None:
        url = client.FinalizeTransferUrl(url)
    self._Initialize(http, url)
    if self.auto_transfer:
        return self.StreamInChunks()
    return http_response