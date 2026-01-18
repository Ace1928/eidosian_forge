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
def PutToken(self, key, value):
    """Serializes the value to the key's filename.

    To ensure that written tokens aren't leaked to a different users, we
     a) unlink an existing cache file, if any (to ensure we don't fall victim
        to symlink attacks and the like),
     b) create a new file with O_CREAT | O_EXCL (to ensure nobody is trying to
        race us)
     If either of these steps fail, we simply give up (but log a warning). Not
     caching access tokens is not catastrophic, and failure to create a file
     can happen for either of the following reasons:
      - someone is attacking us as above, in which case we want to default to
        safe operation (not write the token);
      - another legitimate process is racing us; in this case one of the two
        will win and write the access token, which is fine;
      - we don't have permission to remove the old file or write to the
        specified directory, in which case we can't recover

    Args:
      key: the hash key to store.
      value: the access_token value to serialize.
    """
    cache_file = self.CacheFileName(key)
    LOG.debug('FileSystemTokenCache.PutToken: key=%s, cache_file=%s', key, cache_file)
    try:
        os.unlink(cache_file)
    except:
        pass
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, 'O_NOINHERIT'):
        flags |= os.O_NOINHERIT
    if hasattr(os, 'O_BINARY'):
        flags |= os.O_BINARY
    try:
        fd = os.open(cache_file, flags, 384)
    except (OSError, IOError) as e:
        LOG.warning('FileSystemTokenCache.PutToken: Failed to create cache file %s: %s', cache_file, e)
        return
    f = os.fdopen(fd, 'w+b')
    serialized = value.Serialize()
    if isinstance(serialized, six.text_type):
        serialized = serialized.encode('utf-8')
    f.write(six.ensure_binary(serialized))
    f.close()