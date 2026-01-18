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
def WildcardIterator(self, url_string):
    """Helper to instantiate gslib.WildcardIterator.

    Args are same as gslib.WildcardIterator interface, but this method fills
    in most of the values from instance state.

    Args:
      url_string: URL string naming wildcard objects to iterate.

    Returns:
      Wildcard iterator over URL string.
    """
    return gslib.wildcard_iterator.CreateWildcardIterator(url_string, self.gsutil_api, all_versions=self.all_versions, project_id=self.project_id, ignore_symlinks=self.ignore_symlinks, logger=self.logger)