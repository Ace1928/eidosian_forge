from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA)
class PrivateCaGa(base.Group):
    """Manage private Certificate Authorities on Google Cloud.

  The privateca command group lets you create and manage private certificate
  authorities and certificates.
  """
    category = base.IDENTITY_AND_SECURITY_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args
        base.DisableUserProjectQuota()