from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class SimulateBeta(base.Group):
    """Simulate changes to Organization Policies."""
    detailed_help = {'DESCRIPTION': '          Simulate Org policies.\n\n          More information can be found here:\n          https://cloud.google.com/iam/docs.\n      '}