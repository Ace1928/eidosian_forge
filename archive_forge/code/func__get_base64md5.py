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
def _get_base64md5(self):
    if 'md5' in self.local_hashes and self.local_hashes['md5']:
        md5 = self.local_hashes['md5']
        if not isinstance(md5, bytes):
            md5 = md5.encode('utf-8')
        return binascii.b2a_base64(md5).decode('utf-8').rstrip('\n')