from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
@base.Hidden
class SimulatorBeta(base.Group):
    """Understand access permission impacts before IAM policy change deployment.

  Commands for analyzing access permission impacts before proposed IAM policy
  changes are deployed.
  """

    def Filter(self, context, args):
        """Enables User-Project override for this surface."""
        base.EnableUserProjectQuota()