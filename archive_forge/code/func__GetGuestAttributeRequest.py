from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def _GetGuestAttributeRequest(holder, instance_ref, variable_key):
    req = holder.client.messages.ComputeInstancesGetGuestAttributesRequest(instance=instance_ref.Name(), project=instance_ref.project, variableKey=variable_key, zone=instance_ref.zone)
    return (holder.client.apitools_client.instances, 'GetGuestAttributes', req)