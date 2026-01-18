from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
def AddFieldToUpdateMask(field, patch_request):
    """Adds name of field to update mask."""
    update_mask = patch_request.updateMask
    if update_mask:
        if update_mask.count(field) == 0:
            patch_request.updateMask = update_mask + ',' + field
    else:
        patch_request.updateMask = field
    return patch_request