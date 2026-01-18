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
def WriteToFile(self, filename):
    """Writes out the cache to the given file."""
    json_str = json.dumps({'prefix': self.prefix, 'results': self.results, 'partial-results': self.partial_results, 'timestamp': self.timestamp})
    try:
        with open(filename, 'w') as fp:
            fp.write(json_str)
    except IOError:
        pass