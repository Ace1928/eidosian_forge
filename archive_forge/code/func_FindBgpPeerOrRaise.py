from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import routers_utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
import six
def FindBgpPeerOrRaise(resource, peer_name):
    """Searches for and returns a BGP peer from within a router resource.

  Args:
    resource: The router resource to find the peer for.
    peer_name: The name of the peer to find.

  Returns:
    A reference to the specified peer, if found.

  Raises:
    PeerNotFoundError: If the specified peer was not found in the router.
  """
    for peer in resource.bgpPeers:
        if peer.name == peer_name:
            return peer
    raise PeerNotFoundError(peer_name)