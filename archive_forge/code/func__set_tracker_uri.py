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
def _set_tracker_uri(self, uri):
    """
        Called when we start a new resumable upload or get a new tracker
        URI for the upload. Saves URI and resets upload state.

        Raises InvalidUriError if URI is syntactically invalid.
        """
    parse_result = urlparse.urlparse(uri)
    if parse_result.scheme.lower() not in ['http', 'https'] or not parse_result.netloc:
        raise InvalidUriError('Invalid tracker URI (%s)' % uri)
    self.tracker_uri = uri
    self.tracker_uri_host = parse_result.netloc
    self.tracker_uri_path = '%s?%s' % (parse_result.path, parse_result.query)
    self.server_has_bytes = 0