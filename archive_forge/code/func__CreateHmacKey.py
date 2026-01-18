from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.help_provider import CreateHelpText
from gslib.metrics import LogCommandParams
from gslib.project_id import PopulateProjectId
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.text_util import InsistAscii
from gslib.utils import shim_util
def _CreateHmacKey(self, thread_state=None):
    """Creates HMAC key for a service account."""
    if self.args:
        self.service_account_email = self.args[0]
    else:
        err_msg = '%s %s requires a service account to be specified as the last argument.\n%s'
        raise CommandException(err_msg % (self.command_name, self.action_subcommand, _CREATE_SYNOPSIS))
    gsutil_api = GetCloudApiInstance(self, thread_state=thread_state)
    response = gsutil_api.CreateHmacKey(self.project_id, self.service_account_email, provider='gs')
    print('%-12s %s' % ('Access ID:', response.metadata.accessId))
    print('%-12s %s' % ('Secret:', response.secret))