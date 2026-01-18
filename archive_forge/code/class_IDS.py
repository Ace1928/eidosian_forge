from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class IDS(base.Group):
    """Manage Cloud IDS.

  More information on Cloud IDS Endpoints can be found at
  https://cloud.google.com/cloud-ids
  """
    category = base.NETWORKING_CATEGORY