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
def RefreshResumableUploadState(self):
    """Talk to the server and refresh the state of this resumable upload.

        Returns:
          Response if the upload is complete.
        """
    if self.strategy != RESUMABLE_UPLOAD:
        return
    self.EnsureInitialized()
    refresh_request = http_wrapper.Request(url=self.url, http_method='PUT', headers={'Content-Range': 'bytes */*'})
    refresh_response = http_wrapper.MakeRequest(self.http, refresh_request, redirections=0, retries=self.num_retries)
    range_header = self._GetRangeHeaderFromResponse(refresh_response)
    if refresh_response.status_code in (http_client.OK, http_client.CREATED):
        self.__complete = True
        self.__progress = self.total_size
        self.stream.seek(self.progress)
        self.__final_response = refresh_response
    elif refresh_response.status_code == http_wrapper.RESUME_INCOMPLETE:
        if range_header is None:
            self.__progress = 0
        else:
            self.__progress = self.__GetLastByte(range_header) + 1
        self.stream.seek(self.progress)
    else:
        raise exceptions.HttpError.FromResponse(refresh_response)