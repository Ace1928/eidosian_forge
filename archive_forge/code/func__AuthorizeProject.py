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
@Retry(ServiceException, tries=3, timeout_secs=1)
def _AuthorizeProject(self, project_id, kms_key):
    """Authorizes a project's service account to be used with a KMS key.

    Authorizes the Cloud Storage-owned service account for project_id to be used
    with kms_key.

    Args:
      project_id: (str) Project id string (not number).
      kms_key: (str) Fully qualified resource name for the KMS key.

    Returns:
      (str, bool) A 2-tuple consisting of:
      1) The email address for the service account associated with the project,
         which is authorized to encrypt/decrypt with the specified key.
      2) A bool value - True if we had to grant the service account permission
         to encrypt/decrypt with the given key; False if the required permission
         was already present.
    """
    service_account = self.gsutil_api.GetProjectServiceAccount(project_id, provider='gs').email_address
    kms_api = KmsApi(logger=self.logger)
    self.logger.debug('Getting IAM policy for %s', kms_key)
    try:
        policy = kms_api.GetKeyIamPolicy(kms_key)
        self.logger.debug('Current policy is %s', policy)
        added_new_binding = False
        binding = Binding(role='roles/cloudkms.cryptoKeyEncrypterDecrypter', members=['serviceAccount:%s' % service_account])
        if binding not in policy.bindings:
            policy.bindings.append(binding)
            kms_api.SetKeyIamPolicy(kms_key, policy)
            added_new_binding = True
        return (service_account, added_new_binding)
    except AccessDeniedException:
        if self.warn_on_key_authorize_failure:
            text_util.print_to_fd('\n'.join(textwrap.wrap('Warning: Check that your Cloud Platform project\'s service account has the "cloudkms.cryptoKeyEncrypterDecrypter" role for the specified key. Without this role, you may not be able to encrypt or decrypt objects using the key which will prevent you from uploading or downloading objects.')))
            return (service_account, False)
        else:
            raise