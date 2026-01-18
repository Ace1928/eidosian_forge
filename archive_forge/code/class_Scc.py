from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Scc(base.Group):
    """Manage Cloud SCC resources.

  Commands for managing Google Cloud SCC (Security Command Center) resources.
  """
    category = base.SECURITY_CATEGORY

    def Filter(self, context, args):
        del context, args