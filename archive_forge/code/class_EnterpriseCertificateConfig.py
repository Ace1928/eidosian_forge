from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class EnterpriseCertificateConfig(base.Group):
    """Create enterprise certificate configurations.

  The {command} group lets you create enterprise certificate configurations.
  This configuration will be used by gcloud to communicate with the
  enterprise-certificate-proxy.

  See more details at `gcloud topic client-certificate`.
  """