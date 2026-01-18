from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
def _CreateSetNameRequest(self, client, instance_ref, name):
    return (client.apitools_client.instances, 'SetName', client.messages.ComputeInstancesSetNameRequest(instancesSetNameRequest=client.messages.InstancesSetNameRequest(name=name, currentName=instance_ref.Name()), **instance_ref.AsDict()))