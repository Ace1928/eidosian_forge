from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Deprecate(is_removed=True, warning='This command is deprecated. Please use `gcloud eventarc` instead.', error='This command has been removed. Please use `gcloud eventarc` instead.')
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class EventarcBeta(Eventarc):
    """Manage Eventarc resources."""