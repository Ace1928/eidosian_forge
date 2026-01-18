from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import copy
import json
from apitools.base.py import encoding_helper
from apitools.base.py import transfer
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage import retry_util
from googlecloudsdk.api_lib.storage.gcs_json import metadata_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import scaled_integer
import six
def _copy_acl_from_source_if_source_is_a_cloud_object_and_preserve_acl_is_true(self, destination_metadata):
    if isinstance(self._source_resource, resource_reference.ObjectResource) and self._request_config.resource_args.preserve_acl:
        destination_metadata.acl = copy.deepcopy(self._source_resource.acl)