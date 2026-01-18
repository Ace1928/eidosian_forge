from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import fnmatch
import heapq
import os
import pathlib
import re
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors as api_errors
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import errors as command_errors
from googlecloudsdk.command_lib.storage import folder_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import log
from googlecloudsdk.core.util import debug_output
import six
def _split_on_wildcard(string):
    """Split the string into two such that first part does not have any wildcard.

  Args:
    string (str): The string to be split.

  Returns:
    A 2-tuple where first part doesn't have any wildcard, and second part does
    have a wildcard. If wildcard is not found, the second part is empty.
    If string starts with a wildcard then first part is empty.
    For example:
      _split_on_wildcard('a/b/c/d*e/f/*.txt') => ('a/b/c/d', '*e/f/*.txt')
      _split_on_wildcard('*e/f/*.txt') => ('', '*e/f/*.txt')
      _split_on_wildcard('a/b/c/d') => ('a/b/c/d', '')
  """
    match = WILDCARD_REGEX.search(string)
    if match is None:
        return (string, '')
    first_wildcard_idx = match.start()
    prefix = string[:first_wildcard_idx]
    wildcard_str = string[first_wildcard_idx:]
    return (prefix, wildcard_str)