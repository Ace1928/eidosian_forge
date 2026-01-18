from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class PublicCa(base.Group):
    """Manage accounts for Google Trust Services' Certificate Authority.

  Publicca command group lets you create an external account key used for
  binding to an Automatic Certificate Management Environment (ACME) account.
  """
    category = base.IDENTITY_AND_SECURITY_CATEGORY

    def Filter(self, context, args):
        del context, args
        base.DisableUserProjectQuota()