from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Views(base.Group):
    """Manage log views.

  Commands for managing views. A log view represents a subset of the log entries
  in a Cloud Logging log bucket.
  """