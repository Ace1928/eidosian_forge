from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA)
@base.Hidden
class TroubleshootBeta(base.Group):
    """Troubleshoot IAM policies."""
    detailed_help = {'DESCRIPTION': '          Troubleshoot IAM policies.\n\n          More information can be found here:\n          https://cloud.google.com/iam/docs/troubleshooting-access\n      '}