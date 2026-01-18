from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import errno
import random
import re
import socket
import time
import six
from six.moves import urllib
from six.moves import http_client
from boto import config
from boto import UserAgent
from boto.connection import AWSAuthConnection
from boto.exception import ResumableTransferDisposition
from boto.exception import ResumableUploadException
from gslib.exception import InvalidUrlError
from gslib.utils.boto_util import GetMaxRetryDelay
from gslib.utils.boto_util import GetNumRetries
from gslib.utils.constants import XML_PROGRESS_CALLBACKS
from gslib.utils.constants import UTF8
def SendFile(self, key, fp, size, headers, canned_acl=None, cb=None, num_cb=XML_PROGRESS_CALLBACKS):
    """Upload a file to a key into a bucket on GS, resumable upload protocol.

    Args:
      key: `boto.s3.key.Key` or subclass representing the upload destination.
      fp: File pointer to upload
      size: Size of the file to upload.
      headers: The headers to pass along with the PUT request
      canned_acl: Optional canned ACL to apply to object.
      cb: Callback function that will be called to report progress on
          the upload.  The callback should accept two integer parameters, the
          first representing the number of bytes that have been successfully
          transmitted to GS, and the second representing the total number of
          bytes that need to be transmitted.
      num_cb: (optional) If a callback is specified with the cb parameter, this
              parameter determines the granularity of the callback by defining
              the maximum number of times the callback will be called during the
              file transfer. Providing a negative integer will cause your
              callback to be called with each buffer read.

    Raises:
      ResumableUploadException if a problem occurs during the transfer.
    """
    if not headers:
        headers = {}
    content_type = 'Content-Type'
    if content_type in headers and headers[content_type] is None:
        del headers[content_type]
    if canned_acl:
        headers[key.provider.acl_header] = canned_acl
    headers['User-Agent'] = UserAgent
    file_length = size
    debug = key.bucket.connection.debug
    if self.num_retries is None:
        self.num_retries = GetNumRetries()
    self.progress_less_iterations = 0
    while True:
        service_had_bytes_before_attempt = self.service_has_bytes
        try:
            _, self.generation, self.metageneration = self._AttemptResumableUpload(key, fp, file_length, headers, cb, num_cb)
            key.generation = self.generation
            if debug >= 1:
                self.logger.debug('Resumable upload complete.')
            return
        except self.RETRYABLE_EXCEPTIONS as e:
            if debug >= 1:
                self.logger.debug('Caught exception (%s)', e.__repr__())
            if isinstance(e, IOError) and e.errno == errno.EPIPE:
                key.bucket.connection.connection.close()
        except ResumableUploadException as e:
            self.HandleResumableUploadException(e, debug)
        self.TrackProgressLessIterations(service_had_bytes_before_attempt, debug=debug)