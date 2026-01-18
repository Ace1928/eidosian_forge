from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.forwarding_rules import flags as ilb_flags
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.networks import flags as network_flags
from googlecloudsdk.command_lib.compute.routes import flags
from googlecloudsdk.command_lib.compute.vpn_tunnels import flags as vpn_flags
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class CreateAlphaBeta(Create):
    """Create a new route.

    *{command}* is used to create routes. A route is a rule that
  specifies how certain packets should be handled by the virtual
  network. Routes are associated with virtual machine instances
  by tag, and the set of routes for a particular VM is called
  its routing table. For each packet leaving a virtual machine,
  the system searches that machine's routing table for a single
  best matching route.

  Routes match packets by destination IP address, preferring
  smaller or more specific ranges over larger ones (see
  ``--destination-range''). If there is a tie, the system selects
  the route with the smallest priority value. If there is still
  a tie, it uses the layer 3 and 4 packet headers to
  select just one of the remaining matching routes. The packet
  is then forwarded as specified by ``--next-hop-address'',
  ``--next-hop-instance'', ``--next-hop-vpn-tunnel'', ``--next-hop-gateway'',
  or ``--next-hop-ilb'' of the winning route. Packets that do
  not match any route in the sending virtual machine routing
  table will be dropped.

  Exactly one of ``--next-hop-address'', ``--next-hop-gateway'',
  ``--next-hop-vpn-tunnel'', ``--next-hop-instance'', or ``--next-hop-ilb''
  must be provided with this command.

  ## EXAMPLES

  To create a route with the name 'route-name' with destination range
  '0.0.0.0/0' and with next hop gateway 'default-internet-gateway', run:

    $ {command} route-name \\
      --destination-range=0.0.0.0/0 \\
      --next-hop-gateway=default-internet-gateway

  """