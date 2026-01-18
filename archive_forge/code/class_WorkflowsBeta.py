from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class WorkflowsBeta(base.Group):
    """Manage your Cloud Workflows resources."""
    category = base.TOOLS_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args