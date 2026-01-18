from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.instance_groups import flags
def MakeDiskDeviceNullEntryForDisablingInPatch(client, device_name):
    return client.messages.StatefulPolicyPreservedState.DisksValue.AdditionalProperty(key=device_name, value=None)