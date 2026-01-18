from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import threading
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import gce_cache
from googlecloudsdk.core.credentials import gce_read
from googlecloudsdk.core.util import retry
from six.moves import urllib
def _Wrapper(f):

    def Inner(self, *args, **kwargs):
        try:
            return f(self, *args, **kwargs)
        except CannotConnectToMetadataServerException:
            with _metadata_lock:
                self.connected = gce_cache.ForceCacheRefresh()
            return [] if return_list else None
    return Inner