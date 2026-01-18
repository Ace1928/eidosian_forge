from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils as mig_utils
from googlecloudsdk.api_lib.compute.instance_groups.managed import autoscalers as autoscalers_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
def _SendPatchRequest(self, args, client, autoscalers_client, igm_ref, new_autoscaler):
    if args.IsSpecified('clear_scale_in_control'):
        with client.apitools_client.IncludeFields(['autoscalingPolicy.scaleInControl']):
            return autoscalers_client.Patch(igm_ref, new_autoscaler)
    elif self.clear_scale_down and args.IsSpecified('clear_scale_down_control'):
        with client.apitools_client.IncludeFields(['autoscalingPolicy.scaleDownControl']):
            return autoscalers_client.Patch(igm_ref, new_autoscaler)
    else:
        return autoscalers_client.Patch(igm_ref, new_autoscaler)