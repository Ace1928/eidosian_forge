from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class ConnectionProfiles(base.Group):
    """Manage Database Migration Service connection profiles.

  Commands for managing Database Migration Service connection profiles.
  """