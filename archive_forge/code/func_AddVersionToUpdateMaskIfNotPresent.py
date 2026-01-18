from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.iam import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
def AddVersionToUpdateMaskIfNotPresent(update_mask_path):
    """Add ',version' to update_mask if it is not present."""

    def Process(ref, args, request):
        """The implementation of Process for the hook."""
        del ref, args
        update_mask = arg_utils.GetFieldValueFromMessage(request, update_mask_path)
        if 'version' not in update_mask:
            if update_mask is None:
                update_mask = 'version'
            else:
                update_mask += ',version'
        arg_utils.SetFieldInMessage(request, update_mask_path, update_mask)
        return request
    return Process