from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class Ssl(base.Group):
    """Provide commands for managing SSL certificates of Cloud SQL instances.

  Provide commands for managing client certs and server CA certs of Cloud SQL
  instances.
  """
    category = base.DATABASES_CATEGORY