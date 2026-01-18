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
def ConfigureRequest(self, upload_config, http_request, url_builder):
    """Configure the request and url for this upload."""
    if self.total_size and upload_config.max_size and (self.total_size > upload_config.max_size):
        raise exceptions.InvalidUserInputError('Upload too big: %s larger than max size %s' % (self.total_size, upload_config.max_size))
    if not util.AcceptableMimeType(upload_config.accept, self.mime_type):
        raise exceptions.InvalidUserInputError('MIME type %s does not match any accepted MIME ranges %s' % (self.mime_type, upload_config.accept))
    self.__SetDefaultUploadStrategy(upload_config, http_request)
    if self.strategy == SIMPLE_UPLOAD:
        url_builder.relative_path = upload_config.simple_path
        if http_request.body:
            url_builder.query_params['uploadType'] = 'multipart'
            self.__ConfigureMultipartRequest(http_request)
        else:
            url_builder.query_params['uploadType'] = 'media'
            self.__ConfigureMediaRequest(http_request)
        if self.__gzip_encoded:
            http_request.headers['Content-Encoding'] = 'gzip'
            http_request.body = compression.CompressStream(six.BytesIO(http_request.body))[0].read()
    else:
        url_builder.relative_path = upload_config.resumable_path
        url_builder.query_params['uploadType'] = 'resumable'
        self.__ConfigureResumableRequest(http_request)