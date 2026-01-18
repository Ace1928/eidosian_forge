from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA)
class GSuiteAddOns(base.Group):
    """Manage Google Workspace Add-ons resources.

  Commands for managing Google Workspace Add-ons resources.
  """
    category = base.UNCATEGORIZED_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args