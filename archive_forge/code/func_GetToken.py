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
def GetToken(self, key):
    """Returns a deserialized access token from the key's filename."""
    value = None
    cache_file = self.CacheFileName(key)
    try:
        f = open(cache_file)
        value = AccessToken.UnSerialize(f.read())
        f.close()
    except (IOError, OSError) as e:
        if e.errno != errno.ENOENT:
            LOG.warning('FileSystemTokenCache.GetToken: Failed to read cache file %s: %s', cache_file, e)
    except Exception as e:
        LOG.warning('FileSystemTokenCache.GetToken: Failed to read cache file %s (possibly corrupted): %s', cache_file, e)
    LOG.debug('FileSystemTokenCache.GetToken: key=%s%s present (cache_file=%s)', key, ' not' if value is None else '', cache_file)
    return value