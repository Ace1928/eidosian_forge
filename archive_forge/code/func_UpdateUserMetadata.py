from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_io
def UpdateUserMetadata(unused_instance_ref, args, patch_request):
    """Hook to update deny user metadata to the update mask of the request."""
    if args.IsSpecified('add_viewer_users') or args.IsSpecified('add_standard_users') or args.IsSpecified('add_developer_users'):
        patch_request = AddFieldToUpdateMask('user_metadata', patch_request)
    return patch_request