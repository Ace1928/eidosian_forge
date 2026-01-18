from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Schedules(base.Group):
    """Manage the backup schedules for a Cloud Firestore Database."""