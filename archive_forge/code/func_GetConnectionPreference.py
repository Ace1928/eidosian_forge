from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.network_attachments import flags
from googlecloudsdk.command_lib.compute.networks.subnets import flags as subnetwork_flags
def GetConnectionPreference(args, messages):
    """Get connection preference of the network attachment."""
    if args.connection_preference == 'ACCEPT_AUTOMATIC':
        return messages.NetworkAttachment.ConnectionPreferenceValueValuesEnum.ACCEPT_AUTOMATIC
    if args.connection_preference == 'ACCEPT_MANUAL':
        return messages.NetworkAttachment.ConnectionPreferenceValueValuesEnum.ACCEPT_MANUAL
    return None