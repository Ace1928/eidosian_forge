from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_io
def UpdateDenyMaintenancePeriod(unused_instance_ref, args, patch_request):
    """Hook to update deny maintenance period to the update mask of the request."""
    if args.IsSpecified('deny_maintenance_period_start_date') or args.IsSpecified('deny_maintenance_period_end_date') or args.IsSpecified('deny_maintenance_period_time'):
        patch_request = AddFieldToUpdateMask('deny_maintenance_period', patch_request)
    return patch_request