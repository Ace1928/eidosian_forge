from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.filestore import filestore_client
from googlecloudsdk.command_lib.filestore import update_util
from googlecloudsdk.command_lib.filestore import util
from googlecloudsdk.core import properties
def GetExistingSnapshot(ref, args, patch_request):
    """Fetch existing Filestore instance to update and add it to Patch request."""
    del ref
    resource_ref = GetResourceRef(args)
    api_version = util.GetApiVersionFromArgs(args)
    client = filestore_client.FilestoreClient(api_version)
    orig_snapshot = client.GetInstanceSnapshot(resource_ref)
    patch_request.snapshot = orig_snapshot
    return patch_request