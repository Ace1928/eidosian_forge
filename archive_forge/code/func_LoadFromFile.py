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