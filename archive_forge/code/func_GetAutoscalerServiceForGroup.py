from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
def GetAutoscalerServiceForGroup(self, client, group_ref):
    if _IsZonalGroup(group_ref):
        return client.apitools_client.autoscalers
    else:
        return client.apitools_client.regionAutoscalers