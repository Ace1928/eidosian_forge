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
def DefaultAccount(self):
    """Get the default service account for the host GCE instance.

    Fetches GOOGLE_GCE_METADATA_DEFAULT_ACCOUNT_URI and returns its contents.

    Raises:
      CannotConnectToMetadataServerException: If the metadata server
          cannot be reached.
      MetadataServerException: If there is a problem communicating with the
          metadata server.

    Returns:
      str, The email address for the default service account. None if not on a
          GCE VM, or if there are no service accounts associated with this VM.
    """
    if not self.connected:
        return None
    account = _ReadNoProxyWithCleanFailures(gce_read.GOOGLE_GCE_METADATA_DEFAULT_ACCOUNT_URI, http_errors_to_ignore=(404,))
    if account == CLOUDTOP_COMMON_SERVICE_ACCOUNT:
        return None
    return account