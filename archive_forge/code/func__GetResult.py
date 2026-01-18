from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import filter_rewrite
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.disks import flags as disks_flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _GetResult(compute_client, request, operation_group_id, parent_errors):
    """Requests operations with group id and parses them as an output."""
    is_zonal = hasattr(request, 'zone')
    scope_name = request.zone if is_zonal else request.region
    operations_response, errors = _GetOperations(compute_client, request.project, operation_group_id, scope_name, is_zonal)
    result = {'operationGroupId': operation_group_id, 'createdDisksCount': 0}
    if not parent_errors and (not errors):

        def IsPerDiskOperation(op):
            return op.operationType == 'insert' and str(op.status) == 'DONE' and (op.error is None)
        result['createdDisksCount'] = sum(map(IsPerDiskOperation, operations_response))
    return result