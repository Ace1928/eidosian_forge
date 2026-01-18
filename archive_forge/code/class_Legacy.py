from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Legacy(base.Group):
    """Manages application legacy connectors and connections.

     The gcloud beyondcorp command group lets you secure non-gcp application by
     managing legacy connectors and connections.

     BeyondCorp Enterprise offers a zero trust solution that enables
     secure access with integrated threat and data protection.The solution
     enables secure access to both Google Cloud Platform and on-prem hosted
     apps. For remote apps that are not deployed in Google Cloud Platform,
     BeyondCorp Enterprise's App connector provides simplified
     connectivity and app publishing experience.


     More information on Beyondcorp can be found here:
     https://cloud.google.com/beyondcorp
  """
    category = base.SECURITY_CATEGORY

    def Filter(self, context, args):
        base.DisableUserProjectQuota()