from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class CmekSettings(base.Group):
    """Manages the customer-managed encryption key (CMEK) settings for the Cloud Logging Logs Router."""