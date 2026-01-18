from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.active_directory import util
from googlecloudsdk.command_lib.util.args import labels_util
def UpdatePatchRequest(backup_ref, unused_args, patch_request):
    """Fetch existing AD domain backup to update and add it to Patch request."""
    if patch_request is None:
        return None
    patch_request.backup = GetExistingBackup(backup_ref)
    return patch_request