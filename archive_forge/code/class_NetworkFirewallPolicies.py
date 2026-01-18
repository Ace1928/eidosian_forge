from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class NetworkFirewallPolicies(base.Group):
    """Manage Compute Engine network firewall policies.

  Manage Compute Engine network firewall policies. Network
  firewall policies are used to control incoming/outgoing traffic.
  """
    category = base.COMPUTE_CATEGORY