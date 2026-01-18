from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class Fields(base.Group):
    """Manage single-field indexes for Cloud Firestore.

  Changes here apply to index settings for individual fields, and won't affect
  any composite indexes using those fields.
  """
    pass