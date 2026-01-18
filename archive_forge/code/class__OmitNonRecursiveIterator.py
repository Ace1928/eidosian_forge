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
class _OmitNonRecursiveIterator(object):
    """Iterator wrapper for that omits certain values for non-recursive requests.

  This iterates over tuples of (names_container, BucketListingReference) and
  omits directories, prefixes, and buckets from non-recurisve requests
  so that we can properly calculate whether the source URL expands to multiple
  URLs.

  For example, if we have a bucket containing two objects: bucket/foo and
  bucket/foo/bar and we do a non-recursive iteration, only bucket/foo will be
  yielded.
  """

    def __init__(self, tuple_iter, recursion_requested, command_name, cmd_supports_recursion, logger):
        """Instanties the iterator.

    Args:
      tuple_iter: Iterator over names_container, BucketListingReference
                  from step 2 in the NameExpansionIterator
      recursion_requested: If false, omit buckets, dirs, and subdirs
      command_name: Command name for user messages
      cmd_supports_recursion: Command recursion support for user messages
      logger: Log object for user messages
    """
        self.tuple_iter = tuple_iter
        self.recursion_requested = recursion_requested
        self.command_name = command_name
        self.cmd_supports_recursion = cmd_supports_recursion
        self.logger = logger

    def __iter__(self):
        for names_container, blr in self.tuple_iter:
            if not self.recursion_requested and (not blr.IsObject()):
                expanded_url = StorageUrlFromString(blr.url_string)
                if expanded_url.IsFileUrl():
                    desc = 'directory'
                else:
                    desc = blr.type_name
                if self.cmd_supports_recursion:
                    self.logger.info('Omitting %s "%s". (Did you mean to do %s -r?)', desc, blr.url_string, self.command_name)
                else:
                    self.logger.info('Omitting %s "%s".', desc, blr.url_string)
            else:
                yield (names_container, blr)