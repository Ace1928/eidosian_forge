from __future__ import print_function
import email.utils
import errno
import hashlib
import mimetypes
import os
import re
import base64
import binascii
import math
from hashlib import md5
import boto.utils
from boto.compat import BytesIO, six, urllib, encodebytes
from boto.exception import BotoClientError
from boto.exception import StorageDataError
from boto.exception import PleaseRetryException
from boto.exception import ResumableDownloadException
from boto.exception import ResumableTransferDisposition
from boto.provider import Provider
from boto.s3.keyfile import KeyFile
from boto.s3.user import User
from boto import UserAgent
import boto.utils
from boto.utils import compute_md5, compute_hash
from boto.utils import find_matching_headers
from boto.utils import merge_headers_by_name
from boto.utils import print_to_fd
def _set_base64md5(self, value):
    if value:
        if not isinstance(value, six.string_types):
            value = value.decode('utf-8')
        self.local_hashes['md5'] = binascii.a2b_base64(value)
    elif 'md5' in self.local_hashes:
        del self.local_hashes['md5']