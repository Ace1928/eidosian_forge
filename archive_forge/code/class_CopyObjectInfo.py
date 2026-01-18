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
class CopyObjectInfo(object):
    """Represents the information needed for copying a single object.
  """

    def __init__(self, name_expansion_result, exp_dst_url, have_existing_dst_container):
        """Instantiates the object info from name expansion result and destination.

    Args:
      name_expansion_result: StorageUrl that was being expanded.
      exp_dst_url: StorageUrl of the destination.
      have_existing_dst_container: Whether exp_url names an existing directory,
          bucket, or bucket subdirectory.
    """
        self.source_storage_url = name_expansion_result.source_storage_url
        self.is_multi_source_request = name_expansion_result.is_multi_source_request
        self.is_multi_top_level_source_request = name_expansion_result.is_multi_top_level_source_request
        self.names_container = name_expansion_result.names_container
        self.expanded_storage_url = name_expansion_result.expanded_storage_url
        self.expanded_result = name_expansion_result.expanded_result
        self.exp_dst_url = exp_dst_url
        self.have_existing_dst_container = have_existing_dst_container