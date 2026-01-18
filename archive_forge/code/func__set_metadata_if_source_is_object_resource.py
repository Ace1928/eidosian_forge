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
def _set_metadata_if_source_is_object_resource(self, object_metadata):
    """Copies metadata from _source_resource to object_metadata.

    It is copied if _source_resource is an instance of ObjectResource, this is
    in case a daisy chain copy is performed.

    Args:
      object_metadata (gapic_clients.storage_v2.types.storage.Object): Existing
        object metadata.
    """
    if not isinstance(self._source_resource, resource_reference.ObjectResource):
        return
    if not self._source_resource.custom_fields:
        return
    object_metadata.metadata = copy.deepcopy(self._source_resource.custom_fields)