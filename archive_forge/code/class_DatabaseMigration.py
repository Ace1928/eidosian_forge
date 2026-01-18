from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class DatabaseMigration(base.Group):
    """Manage Database Migration Service resources.

  Commands for managing Database Migration Service resources.
  """
    category = base.DATABASES_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args