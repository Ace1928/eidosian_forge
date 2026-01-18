import errno
import os
import random
import re
import socket
import time
from hashlib import md5
import six.moves.http_client as httplib
from six.moves import urllib as urlparse
from boto import config, UserAgent
from boto.connection import AWSAuthConnection
from boto.exception import InvalidUriError
from boto.exception import ResumableTransferDisposition
from boto.exception import ResumableUploadException
from boto.s3.keyfile import KeyFile
def _start_new_resumable_upload(self, key, headers=None):
    """
        Starts a new resumable upload.

        Raises ResumableUploadException if any errors occur.
        """
    conn = key.bucket.connection
    if conn.debug >= 1:
        print('Starting new resumable upload.')
    self.server_has_bytes = 0
    post_headers = {}
    for k in headers:
        if k.lower() == 'content-length':
            raise ResumableUploadException('Attempt to specify Content-Length header (disallowed)', ResumableTransferDisposition.ABORT)
        post_headers[k] = headers[k]
    post_headers[conn.provider.resumable_upload_header] = 'start'
    resp = conn.make_request('POST', key.bucket.name, key.name, post_headers)
    body = resp.read()
    if resp.status in [500, 503]:
        raise ResumableUploadException('Got status %d from attempt to start resumable upload. Will wait/retry' % resp.status, ResumableTransferDisposition.WAIT_BEFORE_RETRY)
    elif resp.status != 200 and resp.status != 201:
        raise ResumableUploadException('Got status %d from attempt to start resumable upload. Aborting' % resp.status, ResumableTransferDisposition.ABORT)
    tracker_uri = resp.getheader('Location')
    if not tracker_uri:
        raise ResumableUploadException('No resumable tracker URI found in resumable initiation POST response (%s)' % body, ResumableTransferDisposition.WAIT_BEFORE_RETRY)
    self._set_tracker_uri(tracker_uri)
    self._save_tracker_uri_to_file()