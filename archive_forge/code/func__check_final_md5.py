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
def _check_final_md5(self, key, etag):
    """
        Checks that etag from server agrees with md5 computed before upload.
        This is important, since the upload could have spanned a number of
        hours and multiple processes (e.g., gsutil runs), and the user could
        change some of the file and not realize they have inconsistent data.
        """
    if key.bucket.connection.debug >= 1:
        print('Checking md5 against etag.')
    if key.md5 != etag.strip('"\''):
        key.open_read()
        key.close()
        key.delete()
        raise ResumableUploadException("File changed during upload: md5 signature doesn't match etag (incorrect uploaded object deleted)", ResumableTransferDisposition.ABORT)