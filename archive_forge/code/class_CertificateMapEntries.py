from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class CertificateMapEntries(base.Group):
    """Manage Certificate Manager certificate map entries.

  Commands for managing certificate map entries.
  """