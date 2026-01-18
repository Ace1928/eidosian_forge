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
def GetCachedResults(self, prefix):
    """Returns the cached results for prefix or None if not in cache."""
    current_time = time.time()
    if current_time - self.timestamp >= TAB_COMPLETE_CACHE_TTL:
        return None
    results = None
    if prefix == self.prefix:
        results = self.results
    elif not self.partial_results and prefix.startswith(self.prefix) and (prefix.count('/') == self.prefix.count('/')):
        results = [x for x in self.results if x.startswith(prefix)]
    if results is not None:
        self.timestamp = time.time()
        return results