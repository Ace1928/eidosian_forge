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
class CloudObjectCompleter(object):
    """Completer object for Cloud URLs."""

    def __init__(self, gsutil_api, bucket_only=False):
        """Instantiates completer for Cloud URLs.

    Args:
      gsutil_api: gsutil Cloud API instance to use.
      bucket_only: Whether the completer should only match buckets.
    """
        self._gsutil_api = gsutil_api
        self._bucket_only = bucket_only

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

    def __call__(self, prefix, **kwargs):
        if not prefix:
            prefix = 'gs://'
        elif IsFileUrlString(prefix):
            return []
        wildcard_url = prefix + '*'
        url = StorageUrlFromString(wildcard_url)
        if self._bucket_only and (not url.IsBucket()):
            return []
        timeout = boto.config.getint('GSUtil', 'tab_completion_timeout', 5)
        if timeout == 0:
            return []
        start_time = time.time()
        cache = TabCompletionCache.LoadFromFile(GetTabCompletionCacheFilename())
        cached_results = cache.GetCachedResults(prefix)
        timing_log_entry_type = ''
        if cached_results is not None:
            results = cached_results
            timing_log_entry_type = ' (from cache)'
        else:
            try:
                results = self._PerformCloudListing(wildcard_url, timeout)
                if self._bucket_only and len(results) == 1:
                    results = [StripOneSlash(results[0])]
                partial_results = len(results) == _TAB_COMPLETE_MAX_RESULTS
                cache.UpdateCache(prefix, results, partial_results)
            except TimeoutError:
                timing_log_entry_type = ' (request timeout)'
                results = []
        cache.WriteToFile(GetTabCompletionCacheFilename())
        end_time = time.time()
        num_results = len(results)
        elapsed_seconds = end_time - start_time
        _WriteTimingLog('%s results%s in %.2fs, %.2f results/second for prefix: %s\n' % (num_results, timing_log_entry_type, elapsed_seconds, num_results / elapsed_seconds, prefix))
        return results