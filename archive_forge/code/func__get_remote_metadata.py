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
def _get_remote_metadata(self, headers=None):
    """
        Extracts metadata from existing URI into a dict, so we can
        overwrite/delete from it to form the new set of metadata to apply to a
        key.
        """
    metadata = {}
    for underscore_name in self._underscore_base_user_settable_fields:
        if hasattr(self, underscore_name):
            value = getattr(self, underscore_name)
            if value:
                field_name = underscore_name.replace('_', '-')
                metadata[field_name.lower()] = value
    prefix = self.provider.metadata_prefix
    for underscore_name in self.metadata:
        field_name = underscore_name.replace('_', '-')
        metadata['%s%s' % (prefix, field_name.lower())] = self.metadata[underscore_name]
    return metadata