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
def _fetch_sub_bucket_resources(self, bucket_name):
    """Fetch all objects for the given bucket that match the URL."""
    needs_further_expansion = contains_wildcard(self._url.object_name) or self._object_state_requires_expansion or self._url.url_string.endswith(self._url.delimiter)
    if not needs_further_expansion:
        direct_query_result = self._try_getting_object_directly(bucket_name)
        if direct_query_result:
            return [direct_query_result]
    return self._expand_object_path(bucket_name)