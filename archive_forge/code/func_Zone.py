from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import threading
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import gce_cache
from googlecloudsdk.core.credentials import gce_read
from googlecloudsdk.core.util import retry
from six.moves import urllib
@_HandleMissingMetadataServer()
def Zone(self):
    """Get the name of the zone containing the current GCE instance.

    Fetches GOOGLE_GCE_METADATA_ZONE_URI, formats it, and returns its contents.

    Raises:
      CannotConnectToMetadataServerException: If the metadata server
          cannot be reached.
      MetadataServerException: If there is a problem communicating with the
          metadata server.

    Returns:
      str, The short name (e.g., us-central1-f) of the zone containing the
          current instance.
      None if not on a GCE VM.
    """
    if not self.connected:
        return None
    zone_path = _ReadNoProxyWithCleanFailures(gce_read.GOOGLE_GCE_METADATA_ZONE_URI)
    return zone_path.split('/')[-1]