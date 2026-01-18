from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.datastream import util
def _GetRoute(self, route_id, args):
    """Returns a route object."""
    route_obj = self._messages.Route(name=route_id, labels={}, displayName=args.display_name, destinationAddress=args.destination_address, destinationPort=args.destination_port)
    return route_obj