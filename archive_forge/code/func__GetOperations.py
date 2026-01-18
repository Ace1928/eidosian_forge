from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import filter_rewrite
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.disks import flags as disks_flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _GetOperations(compute_client, project, operation_group_id, scope_name, is_zonal):
    """Requests operations with group id matching the given one."""
    errors_to_collect = []
    _, operation_filter = filter_rewrite.Rewriter().Rewrite(expression='operationGroupId=' + operation_group_id)
    if is_zonal:
        operations_response = compute_client.MakeRequests([(compute_client.apitools_client.zoneOperations, 'List', compute_client.apitools_client.zoneOperations.GetRequestType('List')(filter=operation_filter, zone=scope_name, project=project))], errors_to_collect=errors_to_collect, log_result=False, always_return_operation=True, no_followup=True)
    else:
        operations_response = compute_client.MakeRequests([(compute_client.apitools_client.regionOperations, 'List', compute_client.apitools_client.regionOperations.GetRequestType('List')(filter=operation_filter, region=scope_name, project=project))], errors_to_collect=errors_to_collect, log_result=False, always_return_operation=True, no_followup=True)
    return (operations_response, errors_to_collect)