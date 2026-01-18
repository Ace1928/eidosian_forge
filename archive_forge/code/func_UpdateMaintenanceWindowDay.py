from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.redis import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
from six.moves import filter  # pylint: disable=redefined-builtin
def UpdateMaintenanceWindowDay(unused_instance_ref, args, patch_request):
    """Hook to update maintenance window day to the update mask of the request."""
    if args.IsSpecified('maintenance_window_day'):
        patch_request = AddFieldToUpdateMask('maintenance_policy', patch_request)
    return patch_request