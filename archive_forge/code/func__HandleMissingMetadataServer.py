from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import threading
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import gce_cache
from googlecloudsdk.core.credentials import gce_read
from googlecloudsdk.core.util import retry
from six.moves import urllib
def _HandleMissingMetadataServer(return_list=False):
    """Handles when the metadata server is missing and resets the caches.

  If you move gcloud from one environment to another, it might still think it
  in on GCE from a previous invocation (which would result in a crash).
  Instead of crashing, we ignore the error and just update the cache.

  Args:
    return_list: True to return [] instead of None as the default empty answer.

  Returns:
    The value the underlying method would return.
  """

    def _Wrapper(f):

        def Inner(self, *args, **kwargs):
            try:
                return f(self, *args, **kwargs)
            except CannotConnectToMetadataServerException:
                with _metadata_lock:
                    self.connected = gce_cache.ForceCacheRefresh()
                return [] if return_list else None
        return Inner
    return _Wrapper