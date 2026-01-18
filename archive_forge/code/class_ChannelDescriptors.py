from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class ChannelDescriptors(base.Group):
    """Read Cloud Monitoring notification channel descriptors."""
    detailed_help = {'DESCRIPTION': '          Manage Monitoring notification channel descriptors.\n\n          More information can be found here:\n          https://cloud.google.com/monitoring/api/v3/\n          https://cloud.google.com/monitoring/api/ref_v3/rest/v3/projects.notificationChannelDescriptors\n          https://cloud.google.com/monitoring/alerts/using-channels-api\n      '}