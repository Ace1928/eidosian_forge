from __future__ import absolute_import
import datetime
import errno
from hashlib import sha1
import json
import logging
import os
import socket
import tempfile
import threading
import boto
import httplib2
import oauth2client.client
import oauth2client.service_account
from google_reauth import reauth_creds
import retry_decorator.retry_decorator
import six
from six import BytesIO
from six.moves import urllib
def CacheKey(self):
    """Computes a cache key.

    The cache key is computed as the SHA1 hash of the refresh token for user
    accounts, or the hash of the gs_service_client_id for service accounts,
    which satisfies the FileSystemTokenCache requirement that cache keys do not
    leak information about token values.

    Returns:
      A hash key.
    """
    h = sha1()
    if isinstance(self.cache_key_base, six.text_type):
        val = self.cache_key_base.encode('utf-8')
    else:
        val = self.cache_key_base
    h.update(val)
    return h.hexdigest()