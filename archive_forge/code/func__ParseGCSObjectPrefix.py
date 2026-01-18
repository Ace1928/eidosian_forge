from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.datastore import admin_api
from googlecloudsdk.api_lib.datastore import operations
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.datastore import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _ParseGCSObjectPrefix(self, resource):
    """Parses a GCS bucket with an optional object prefix.

    Args:
      resource: the user input resource string.
    Returns:
      a tuple of strings containing the GCS bucket and GCS object. The GCS
      object may be None.
    """
    try:
        bucket_ref = resources.REGISTRY.Parse(resource, collection='storage.buckets')
        return (bucket_ref.bucket, None)
    except resources.UserError:
        pass
    object_ref = resources.REGISTRY.Parse(resource, collection='storage.objects')
    return (object_ref.bucket, object_ref.object)