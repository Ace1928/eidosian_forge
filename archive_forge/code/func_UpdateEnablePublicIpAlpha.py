from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_io
def UpdateEnablePublicIpAlpha(unused_instance_ref, args, patch_request):
    """Hook to update public IP to the update mask of the request for alpha."""
    if args.IsSpecified('enable_public_ip'):
        patch_request = AddFieldToUpdateMask('enable_public_ip', patch_request)
    return patch_request