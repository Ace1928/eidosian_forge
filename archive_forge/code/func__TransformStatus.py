from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.vmware.privateconnectionroutes import PrivateConnectionPeeringRoutesClient
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.vmware import flags
from googlecloudsdk.core.resource import resource_projector
def _TransformStatus(direction, imported):
    """Create customized status field based on direction and imported."""
    if imported:
        if direction == 'INCOMING':
            return 'accepted'
        return 'accepted by peer'
    if direction == 'INCOMING':
        return 'rejected by config'
    return 'rejected by peer config'