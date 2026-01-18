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
class FileSystemTokenCache(TokenCache):
    """An implementation of a token cache that persists tokens on disk.

  Each token object in the cache is stored in serialized form in a separate
  file. The cache file's name can be configured via a path pattern that is
  parameterized by the key under which a value is cached and optionally the
  current processes uid as obtained by os.getuid().

  Since file names are generally publicly visible in the system, it is important
  that the cache key does not leak information about the token's value.  If
  client code computes cache keys from token values, a cryptographically strong
  one-way function must be used.
  """

    def __init__(self, path_pattern=None):
        """Creates a FileSystemTokenCache.

    Args:
      path_pattern: Optional string argument to specify the path pattern for
          cache files.  The argument should be a path with format placeholders
          '%(key)s' and optionally '%(uid)s'.  If the argument is omitted, the
          default pattern
            <tmpdir>/oauth2client-tokencache.%(uid)s.%(key)s
          is used, where <tmpdir> is replaced with the system temp dir as
          obtained from tempfile.gettempdir().
    """
        super(FileSystemTokenCache, self).__init__()
        self.path_pattern = path_pattern
        if not path_pattern:
            self.path_pattern = os.path.join(tempfile.gettempdir(), 'oauth2_client-tokencache.%(uid)s.%(key)s')

    def CacheFileName(self, key):
        uid = '_'
        try:
            uid = str(os.getuid())
        except:
            pass
        return self.path_pattern % {'key': key, 'uid': uid}

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