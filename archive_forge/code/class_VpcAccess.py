from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA)
class VpcAccess(base.Group):
    """Manage VPC Access Service resources.

  Commands for managing Google VPC Access Service resources.
  """

    def Filter(self, context, args):
        del context, args
        base.EnableUserProjectQuota()