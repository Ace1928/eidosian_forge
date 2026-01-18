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
def _UpdateHmacKey(self, thread_state=None):
    """Update an HMAC key's state."""
    if not self.state:
        raise CommandException('A state flag must be supplied for %s %s\n%s' % (self.command_name, self.action_subcommand, _UPDATE_SYNOPSIS))
    elif self.state not in _VALID_UPDATE_STATES:
        raise CommandException('The state flag value must be one of %s' % ', '.join(_VALID_UPDATE_STATES))
    if self.args:
        access_id = self.args[0]
    else:
        raise _AccessIdException(self.command_name, self.action_subcommand, _UPDATE_SYNOPSIS)
    gsutil_api = GetCloudApiInstance(self, thread_state=thread_state)
    response = gsutil_api.UpdateHmacKey(self.project_id, access_id, self.state, self.etag, provider='gs')
    print(_KeyMetadataOutput(response))