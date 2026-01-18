from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import threading
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import gce_cache
from googlecloudsdk.core.credentials import gce_read
from googlecloudsdk.core.util import retry
from six.moves import urllib
@_HandleMissingMetadataServer(return_list=True)
def Accounts(self):
    """Get the list of service accounts available from the metadata server.

    Returns:
      [str], The list of accounts. [] if not on a GCE VM.

    Raises:
      CannotConnectToMetadataServerException: If no metadata server is present.
      MetadataServerException: If there is a problem communicating with the
          metadata server.
    """
    if not self.connected:
        return []
    accounts_listing = _ReadNoProxyWithCleanFailures(gce_read.GOOGLE_GCE_METADATA_ACCOUNTS_URI + '/')
    accounts_lines = accounts_listing.split()
    accounts = []
    for account_line in accounts_lines:
        account = account_line.strip('/')
        if account == 'default' or account == CLOUDTOP_COMMON_SERVICE_ACCOUNT:
            continue
        accounts.append(account)
    return accounts