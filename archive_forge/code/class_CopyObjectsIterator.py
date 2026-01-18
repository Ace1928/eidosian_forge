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
class CopyObjectsIterator(six.Iterator):
    """Iterator wrapper for copying objects and keeping track of source URL types.

  This is used in the cp command for copying from multiple source to multiple
  destinations. It takes a list of NameExpansionIteratorDestinationTuple. It
  wraps them and return CopyObjectInfo objects that wraps NameExpansionResult
  with the destination. It's used also for collecting analytics
  PerformanceSummary info, because there may be multiple source URLs and we want
  to know if any of them are file URLs, if any of them are cloud URLs, if any of
  them require daisy chain operations, and if any use different providers. The
  source URL type information will be aggregated at the end of _SequentialApply
  or _ParallelApply.
  """

    def __init__(self, name_expansion_dest_iter, is_daisy_chain):
        """Instantiates the iterator.

    Args:
      name_expansion_dest_iter: NameExpansionIteratorDestinationTuple iterator.
      is_daisy_chain: The -D option in cp might have already been specified, in
          which case we do not need to check again for daisy chain operations.
    """
        self.is_daisy_chain = is_daisy_chain
        self.has_file_src = False
        self.has_cloud_src = False
        self.provider_types = []
        self.name_expansion_dest_iter = name_expansion_dest_iter
        name_expansion_dest_tuple = next(self.name_expansion_dest_iter)
        self.current_expansion_iter = name_expansion_dest_tuple.name_expansion_iter
        self.current_destination = name_expansion_dest_tuple.destination

    def __iter__(self):
        return self

    def __next__(self):
        """Keeps track of URL types as the command iterates over arguments."""
        try:
            name_expansion_result = next(self.current_expansion_iter)
        except StopIteration:
            name_expansion_dest_tuple = next(self.name_expansion_dest_iter)
            self.current_expansion_iter = name_expansion_dest_tuple.name_expansion_iter
            self.current_destination = name_expansion_dest_tuple.destination
            return self.__next__()
        elt = CopyObjectInfo(name_expansion_result, self.current_destination.exp_dst_url, self.current_destination.have_existing_dst_container)
        if not self.has_file_src and elt.source_storage_url.IsFileUrl():
            self.has_file_src = True
        if not self.has_cloud_src and elt.source_storage_url.IsCloudUrl():
            self.has_cloud_src = True
        if self.current_destination.exp_dst_url.IsCloudUrl():
            dst_url_scheme = self.current_destination.exp_dst_url.scheme
        else:
            dst_url_scheme = None
        if not self.is_daisy_chain and dst_url_scheme is not None and elt.source_storage_url.IsCloudUrl() and (elt.source_storage_url.scheme != dst_url_scheme):
            self.is_daisy_chain = True
        if elt.source_storage_url.scheme not in self.provider_types:
            self.provider_types.append(elt.source_storage_url.scheme)
        return elt