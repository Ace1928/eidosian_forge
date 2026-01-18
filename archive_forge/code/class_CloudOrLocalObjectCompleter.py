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
class CloudOrLocalObjectCompleter(object):
    """Completer object for Cloud URLs or local files.

  Invokes the Cloud object completer if the input looks like a Cloud URL and
  falls back to local file completer otherwise.
  """

    def __init__(self, gsutil_api):
        self.cloud_object_completer = CloudObjectCompleter(gsutil_api)
        self.local_object_completer = LocalObjectCompleter()

    def __call__(self, prefix, **kwargs):
        if IsFileUrlString(prefix):
            completer = self.local_object_completer
        else:
            completer = self.cloud_object_completer
        return completer(prefix, **kwargs)