from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class ClientConnector(base.Group):
    """Provides context-aware access to non-web applications (Deprecated).

     BeyondCorp Enterprise is the zero trust solution from Google that provides
     secure access to private applications with integrated threat and data
     protection. BeyondCorp Enterprise uses Chrome to provide secure access for
     all web-based (HTTPS) applications.

     The BeyondCorp Enterprise client connector extends support to non-web
     applications by creating a secure connection to applications running in
     both Google Cloud and non-Google Cloud environments with full context and
     identity aware access.

     More information on Beyondcorp can be found here:
     https://cloud.google.com/beyondcorp-enterprise/docs/client-connector
  """
    category = base.SECURITY_CATEGORY

    def Filter(self, context, args):
        base.DisableUserProjectQuota()