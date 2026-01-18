from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA)
class Transcoder(base.Group):
    """Manage Transcoder jobs and job templates."""
    category = base.SOLUTIONS_CATEGORY

    def Filter(self, context, args):
        del context, args
        base.DisableUserProjectQuota()