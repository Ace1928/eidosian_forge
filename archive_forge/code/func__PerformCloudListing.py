from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import itertools
import json
import threading
import time
import boto
from boto.gs.acl import CannedACLStrings
from gslib.storage_url import IsFileUrlString
from gslib.storage_url import StorageUrlFromString
from gslib.storage_url import StripOneSlash
from gslib.utils.boto_util import GetTabCompletionCacheFilename
from gslib.utils.boto_util import GetTabCompletionLogFilename
from gslib.wildcard_iterator import CreateWildcardIterator
def _PerformCloudListing(self, wildcard_url, timeout):
    """Perform a remote listing request for the given wildcard URL.

    Args:
      wildcard_url: The wildcard URL to list.
      timeout: Time limit for the request.
    Returns:
      Cloud resources matching the given wildcard URL.
    Raises:
      TimeoutError: If the listing does not finish within the timeout.
    """
    request_thread = CloudListingRequestThread(wildcard_url, self._gsutil_api)
    request_thread.start()
    request_thread.join(timeout)
    if request_thread.is_alive():
        import argcomplete
        argcomplete.warn(_TIMEOUT_WARNING % timeout)
        raise TimeoutError()
    results = request_thread.results
    return results