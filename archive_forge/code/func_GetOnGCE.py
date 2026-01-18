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