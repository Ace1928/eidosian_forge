from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class PolicyTroubleshoot(base.Group):
    """Troubleshoot Google Cloud Platform policies.

     Policy Troubleshooter troubleshoots policies for
     Google Cloud Platform resources. Policy Troubleshooter works by
     evaluating the user's current access to a resource.
  """
    category = base.IDENTITY_AND_SECURITY_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args