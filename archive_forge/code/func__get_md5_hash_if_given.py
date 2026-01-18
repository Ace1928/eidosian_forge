from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import copy
import functools
import os
from googlecloudsdk.api_lib.storage import retry_util as storage_retry_util
from googlecloudsdk.api_lib.storage.gcs_grpc import grpc_util
from googlecloudsdk.api_lib.storage.gcs_grpc import metadata_util
from googlecloudsdk.api_lib.storage.gcs_grpc import retry_util
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import scaled_integer
import six
def _get_md5_hash_if_given(self):
    """Returns MD5 hash bytes sequence from resource args if given.

    Returns:
      bytes|None: MD5 hash bytes sequence if MD5 string was given, otherwise
      None.
    """
    if self._request_config.resource_args is not None and self._request_config.resource_args.md5_hash is not None:
        return hash_util.get_bytes_from_base64_string(self._request_config.resource_args.md5_hash)
    return None