from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.filestore import filestore_client
from googlecloudsdk.command_lib.filestore import update_util
from googlecloudsdk.command_lib.filestore import util
from googlecloudsdk.core import properties
def AddDescription(unused_instance_ref, args, patch_request):
    """Adds description to the patch request."""
    return update_util.AddDescription(unused_instance_ref, args, patch_request, update_util.snapshot_feature_name)