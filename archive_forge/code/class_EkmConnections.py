from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.core import resources
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class EkmConnections(base.Group):
    """Create and manage ekm connections.

  Ekm Connections are used to control the connection settings for an
  EXTERNAL_VPC CryptoKey.
  """
    category = base.IDENTITY_AND_SECURITY_CATEGORY

    @staticmethod
    def Args(parser):
        parser.display_info.AddUriFunc(cloudkms_base.MakeGetUriFunc(flags.EKM_CONNECTION_COLLECTION))