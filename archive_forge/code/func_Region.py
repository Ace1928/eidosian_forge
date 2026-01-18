from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import threading
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import gce_cache
from googlecloudsdk.core.credentials import gce_read
from googlecloudsdk.core.util import retry
from six.moves import urllib
def Region(self):
    """Get the name of the region containing the current GCE instance.

    Fetches GOOGLE_GCE_METADATA_ZONE_URI, extracts the region associated
    with the zone, and returns it.  Extraction is based property that
    zone names have form <region>-<zone> (see https://cloud.google.com/
    compute/docs/zones) and an assumption that <zone> contains no hyphens.

    Raises:
      CannotConnectToMetadataServerException: If the metadata server
          cannot be reached.
      MetadataServerException: If there is a problem communicating with the
          metadata server.

    Returns:
      str, The short name (e.g., us-central1) of the region containing the
          current instance.
      None if not on a GCE VM.
    """
    if not self.connected:
        return None
    zone = self.Zone()
    return '-'.join(zone.split('-')[:-1]) if zone else None