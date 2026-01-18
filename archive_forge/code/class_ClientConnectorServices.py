from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class ClientConnectorServices(base.Group):
    """Create and manipulate BeyondCorp client connector services.

  The client connector service is used to define a common configuration for a
  group of client gateways. Client gateways refer to the client connector
  service and are used to control the regions where you want to manage user
  traffic.
    The gateways communicate with the BeyondCorp Enterprise enforcement system
  to enforce context-aware checks. The BeyondCorp Enterprise enforcement system
  uses Identity-Aware Proxy and Access Context Manager, a flexible BeyondCorp
  Enterprise zero trust policy engine.
  """