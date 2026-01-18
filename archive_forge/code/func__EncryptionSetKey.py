from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import getopt
import textwrap
from gslib import metrics
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import ServiceException
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.help_provider import CreateHelpText
from gslib.kms_api import KmsApi
from gslib.project_id import PopulateProjectId
from gslib.third_party.kms_apitools.cloudkms_v1_messages import Binding
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import text_util
from gslib.utils.constants import NO_MAX
from gslib.utils.encryption_helper import ValidateCMEK
from gslib.utils.retry_util import Retry
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
def _EncryptionSetKey(self, bucket_metadata, bucket_url, svc_acct_for_project_num):
    """Sets defaultKmsKeyName on a Cloud Storage bucket.

    Args:
      bucket_metadata: (apitools_messages.Bucket) Metadata for the given bucket.
      bucket_url: (gslib.storage_url.StorageUrl) StorageUrl of the given bucket.
      svc_acct_for_project_num: (Dict[int, str]) Mapping of project numbers to
          their corresponding service account.
    """
    bucket_project_number = bucket_metadata.projectNumber
    try:
        service_account, newly_authorized = (svc_acct_for_project_num[bucket_project_number], False)
    except KeyError:
        service_account, newly_authorized = self._AuthorizeProject(bucket_project_number, self.kms_key)
        svc_acct_for_project_num[bucket_project_number] = service_account
    if newly_authorized:
        text_util.print_to_fd('Authorized service account %s to use key:\n%s' % (service_account, self.kms_key))
    bucket_metadata.encryption = apitools_messages.Bucket.EncryptionValue(defaultKmsKeyName=self.kms_key)
    print('Setting default KMS key for bucket %s...' % str(bucket_url).rstrip('/'))
    self.gsutil_api.PatchBucket(bucket_url.bucket_name, bucket_metadata, fields=['encryption'], provider=bucket_url.scheme)