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
def _get_regex_patterns(self, wildcard_pattern):
    """Returns list of regex patterns derived from the wildcard patterns.

    Args:
      wildcard_pattern (str): A wilcard_pattern to filter the resources.

    Returns:
      List of compiled regex patterns.

    This translates the wildcard_pattern and also creates some additional
    patterns so that we can treat ** in a/b/c/**/d.txt as zero or more folders.
    This means, a/b/c/d.txt will also be returned along with a/b/c/e/f/d.txt.
    """
    wildcard_patterns = [wildcard_pattern]
    if not wildcard_pattern.endswith(storage_url.CLOUD_URL_DELIMITER):
        wildcard_patterns.append(wildcard_pattern + storage_url.CLOUD_URL_DELIMITER)
    if '/**/' in wildcard_pattern:
        updated_pattern = wildcard_pattern.replace('/**/', '/')
        wildcard_patterns.append(updated_pattern)
    else:
        updated_pattern = wildcard_pattern
    for pattern in (wildcard_pattern, updated_pattern):
        if pattern.startswith('**/'):
            wildcard_patterns.append(pattern[3:])
    return [re.compile(fnmatch.translate(p)) for p in wildcard_patterns]