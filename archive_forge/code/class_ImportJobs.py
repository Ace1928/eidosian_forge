from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import flags
class ImportJobs(base.Group):
    """Create and manage import jobs.

  Import jobs can be used to create CryptoKeyVersions using
  pre-existing key material, generated outside of Cloud KMS.
  """
    category = base.IDENTITY_AND_SECURITY_CATEGORY

    @classmethod
    def Args(cls, parser):
        parser.display_info.AddUriFunc(cloudkms_base.MakeGetUriFunc(flags.IMPORT_JOB_COLLECTION))