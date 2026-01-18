from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA)
class AuditLogsProvider(base.Group):
    """Explore provider serviceNames and methodNames for event type `google.cloud.audit.log.v1.written` in Eventarc."""