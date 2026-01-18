from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class ActivePeeringZones(base.Group):
    """Manage your Cloud DNS active-peering-zones.

  Manage your Cloud DNS active-peering-zones.

  ## EXAMPLES

  To list the consumer active-peering-zones targeting a producer network, run:

    $ {command} list --target-network="my-producer-network"

  To revoke a peering connection from the producer by deactivating
  an active-peering-zone, run:

    $ {command} deactivate consumer_zone_id

  """