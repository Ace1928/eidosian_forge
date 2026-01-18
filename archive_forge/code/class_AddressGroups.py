from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class AddressGroups(base.Group):
    """Manage Network Security AddressGroups."""
    category = base.NETWORK_SECURITY_CATEGORY