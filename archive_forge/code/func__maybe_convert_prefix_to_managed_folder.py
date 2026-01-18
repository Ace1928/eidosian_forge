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
def _maybe_convert_prefix_to_managed_folder(self, resource):
    """If resource is a prefix, attempts to convert it to a managed folder."""
    if type(resource) is not resource_reference.PrefixResource or self._managed_folder_setting not in {folder_util.ManagedFolderSetting.LIST_WITH_OBJECTS, folder_util.ManagedFolderSetting.LIST_WITHOUT_OBJECTS} or cloud_api.Capability.MANAGED_FOLDERS not in self._client.capabilities:
        return resource
    try:
        prefix_url = resource.storage_url
        return self._client.get_managed_folder(prefix_url.bucket_name, prefix_url.object_name)
    except api_errors.NotFoundError:
        return resource