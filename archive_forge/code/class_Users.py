from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Users(base.Group):
    """Provide commands for managing AlloyDB users.

  Provide commands for managing AlloyDB users including creating,
  configuring, getting, listing, and deleting users.
  """