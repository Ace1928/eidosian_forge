from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
import logging
import os
import sys
import six
from apitools.base.py import encoding
import gslib
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_GENERIC
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.plurality_checkable_iterator import PluralityCheckableIterator
from gslib.seek_ahead_thread import SeekAheadResult
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
import gslib.wildcard_iterator
from gslib.wildcard_iterator import StorageUrlFromString
class _NonContainerTuplifyIterator(object):
    """Iterator that produces the tuple (False, blr) for each iterated value.

  Used for cases where blr_iter iterates over a set of
  BucketListingRefs known not to name containers.
  """

    def __init__(self, blr_iter):
        """Instantiates iterator.

    Args:
      blr_iter: iterator of BucketListingRef.
    """
        self.blr_iter = blr_iter

    def __iter__(self):
        for blr in self.blr_iter:
            yield (False, blr)