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
class TabCompletionCache(object):
    """Cache for tab completion results."""

    def __init__(self, prefix, results, timestamp, partial_results):
        self.prefix = prefix
        self.results = results
        self.timestamp = timestamp
        self.partial_results = partial_results

    @staticmethod
    def LoadFromFile(filename):
        """Instantiates the cache from a file.

    Args:
      filename: The file to load.
    Returns:
      TabCompletionCache instance with loaded data or an empty cache
          if the file cannot be loaded
    """
        try:
            with open(filename, 'r') as fp:
                cache_dict = json.loads(fp.read())
                prefix = cache_dict['prefix']
                results = cache_dict['results']
                timestamp = cache_dict['timestamp']
                partial_results = cache_dict['partial-results']
        except Exception:
            prefix = None
            results = []
            timestamp = 0
            partial_results = False
        return TabCompletionCache(prefix, results, timestamp, partial_results)

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

    def UpdateCache(self, prefix, results, partial_results):
        """Updates the in-memory cache with the results for the given prefix."""
        self.prefix = prefix
        self.results = results
        self.partial_results = partial_results
        self.timestamp = time.time()

    def WriteToFile(self, filename):
        """Writes out the cache to the given file."""
        json_str = json.dumps({'prefix': self.prefix, 'results': self.results, 'partial-results': self.partial_results, 'timestamp': self.timestamp})
        try:
            with open(filename, 'w') as fp:
                fp.write(json_str)
        except IOError:
            pass