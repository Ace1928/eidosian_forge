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
def _ServiceAccount(self):
    self.CheckArguments()
    if not self.args:
        self.args = ['gs://']
    if self.sub_opts:
        for o, a in self.sub_opts:
            if o == '-p':
                self.project_id = a
    if not self.project_id:
        self.project_id = PopulateProjectId(None)
    self.logger.debug('Checking service account for project %s', self.project_id)
    service_account = self.gsutil_api.GetProjectServiceAccount(self.project_id, provider='gs').email_address
    print(service_account)
    return 0