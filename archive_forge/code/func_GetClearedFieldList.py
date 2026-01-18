from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.backend_services import backend_services_utils
from googlecloudsdk.command_lib.compute.backend_services import flags
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.console import console_io
def GetClearedFieldList(self, backend_service):
    """Retrieves a list of fields to clear for the backend service being inserted.

    Args:
      backend_service: The backend service being inserted.

    Returns:
      The the list of fields to clear for a GA resource.
    """
    cleared_fields = super().GetClearedFieldList(backend_service)
    if backend_service.haPolicy:
        ha_policy = backend_service.haPolicy
        if not ha_policy.fastIPMove:
            cleared_fields.append('haPolicy.fastIPMove')
        if ha_policy.leader:
            leader = ha_policy.leader
            if not leader.backendGroup:
                cleared_fields.append('haPolicy.leader.backendGroup')
            if leader.networkEndpoint:
                if not leader.networkEndpoint.instance:
                    cleared_fields.append('haPolicy.leader.networkEndpoint.instance')
            else:
                cleared_fields.append('haPolicy.leader.networkEndpoint')
        else:
            cleared_fields.append('haPolicy.leader')
    else:
        cleared_fields.append('haPolicy')
    return cleared_fields