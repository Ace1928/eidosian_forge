from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import times
def RefreshUpdateMask(unused_ref, args, req):
    """Refresh the update mask of the updateTableRequest according to the input arguments.

  Args:
    unused_ref: the gcloud resource (unused).
    args: the input arguments.
    req: the original updateTableRequest.

  Returns:
    req: the updateTableRequest with update mask refreshed.
  """
    if args.clear_change_stream_retention_period:
        req = AddFieldToUpdateMask('changeStreamConfig', req)
    if args.change_stream_retention_period:
        req = AddFieldToUpdateMask('changeStreamConfig.retentionPeriod', req)
    if args.enable_automated_backup or args.disable_automated_backup:
        req = AddFieldToUpdateMask('automatedBackupPolicy', req)
    return req