from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import socket
import threading
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import gce_read
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import retry
import six
from six.moves import http_client
from six.moves import urllib_error
class _OnGCECache(object):
    """Logic to check if we're on GCE and cache the result to file or memory.

  Checking if we are on GCE is done by issuing an HTTP request to a GCE server.
  Since HTTP requests are slow, we cache this information. Because every run
  of gcloud is a separate command, the cache is stored in a file in the user's
  gcloud config dir. Because within a gcloud run we might check if we're on GCE
  multiple times, we also cache this information in memory.
  A user can move the gcloud instance to and from a GCE VM, and the GCE server
  can sometimes not respond. Therefore the cache has an age and gets refreshed
  if more than _GCE_CACHE_MAX_AGE passed since it was updated.
  """

    def __init__(self, connected=None, expiration_time=None):
        self.connected = connected
        self.expiration_time = expiration_time
        self.file_lock = threading.Lock()

    def GetOnGCE(self, check_age=True):
        """Check if we are on a GCE machine.

    Checks, in order:
    * in-memory cache
    * on-disk cache
    * metadata server

    If we read from one of these sources, update all of the caches above it in
    the list.

    If check_age is True, then update all caches if the information we have is
    older than _GCE_CACHE_MAX_AGE. In most cases, age should be respected. It
    was added for reporting metrics.

    Args:
      check_age: bool, determines if the cache should be refreshed if more than
        _GCE_CACHE_MAX_AGE time passed since last update.

    Returns:
      bool, if we are on GCE or not.
    """
        on_gce = self._CheckMemory(check_age=check_age)
        if on_gce is not None:
            return on_gce
        self._WriteMemory(*self._CheckDisk())
        on_gce = self._CheckMemory(check_age=check_age)
        if on_gce is not None:
            return on_gce
        return self.CheckServerRefreshAllCaches()

    def CheckServerRefreshAllCaches(self):
        on_gce = self._CheckServerWithRetry()
        self._WriteDisk(on_gce)
        self._WriteMemory(on_gce, time.time() + _GCE_CACHE_MAX_AGE)
        return on_gce

    def _CheckMemory(self, check_age):
        if not check_age:
            return self.connected
        if self.expiration_time and self.expiration_time >= time.time():
            return self.connected
        return None

    def _WriteMemory(self, on_gce, expiration_time):
        self.connected = on_gce
        self.expiration_time = expiration_time

    def _CheckDisk(self):
        """Reads cache from disk."""
        gce_cache_path = config.Paths().GCECachePath()
        with self.file_lock:
            try:
                mtime = os.stat(gce_cache_path).st_mtime
                expiration_time = mtime + _GCE_CACHE_MAX_AGE
                gcecache_file_value = files.ReadFileContents(gce_cache_path)
                return (gcecache_file_value == six.text_type(True), expiration_time)
            except (OSError, IOError, files.Error):
                return (None, None)

    def _WriteDisk(self, on_gce):
        """Updates cache on disk."""
        gce_cache_path = config.Paths().GCECachePath()
        with self.file_lock:
            try:
                files.WriteFileContents(gce_cache_path, six.text_type(on_gce), private=True)
            except (OSError, IOError, files.Error):
                pass

    def _CheckServerWithRetry(self):
        try:
            return self._CheckServer()
        except _POSSIBLE_ERRORS_GCE_METADATA_CONNECTION:
            return False

    @retry.RetryOnException(max_retrials=3, should_retry_if=_ShouldRetryMetadataServerConnection)
    def _CheckServer(self):
        return gce_read.ReadNoProxy(gce_read.GOOGLE_GCE_METADATA_NUMERIC_PROJECT_URI, properties.VALUES.compute.gce_metadata_check_timeout_sec.GetInt()).isdigit()