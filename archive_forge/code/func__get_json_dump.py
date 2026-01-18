from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from apitools.base.py import encoding
from googlecloudsdk.command_lib.storage.resources import full_resource_formatter
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
def _get_json_dump(resource):
    """Formats GCS resource metadata for printing.

  Args:
    resource (GcsBucketResource|GcsObjectResource): Resource object.

  Returns:
    Formatted JSON string for printing.
  """
    return resource_util.configured_json_dumps(collections.OrderedDict([('url', resource.storage_url.url_string), ('type', resource.TYPE_STRING), ('metadata', _json_dump_helper(resource.metadata))]))