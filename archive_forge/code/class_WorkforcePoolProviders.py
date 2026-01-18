from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class WorkforcePoolProviders(base.Group):
    """Create and manage workforce pool providers.

  The {command} group lets you create and manage workforce pool providers.
  """