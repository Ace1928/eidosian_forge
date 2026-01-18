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
class HmacCommand(Command):
    """Implementation of gsutil hmac command."""
    command_spec = Command.CreateCommandSpec('hmac', min_args=1, max_args=8, supported_sub_args='ae:lp:s:u:', file_url_ok=True, urls_start_arg=1, gs_api_support=[ApiSelector.JSON], gs_default_api=ApiSelector.JSON, usage_synopsis=_SYNOPSIS, argparse_arguments={'create': [CommandArgument.MakeZeroOrMoreCloudOrFileURLsArgument()], 'delete': [CommandArgument.MakeZeroOrMoreCloudOrFileURLsArgument()], 'get': [CommandArgument.MakeZeroOrMoreCloudOrFileURLsArgument()], 'list': [CommandArgument.MakeZeroOrMoreCloudOrFileURLsArgument()], 'update': [CommandArgument.MakeZeroOrMoreCloudOrFileURLsArgument()]})
    help_spec = Command.HelpSpec(help_name='hmac', help_name_aliases=[], help_type='command_help', help_one_line_summary='CRUD operations on service account HMAC keys.', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={'create': _create_help_text, 'delete': _delete_help_text, 'get': _get_help_text, 'list': _list_help_text, 'update': _update_help_text})

    def get_gcloud_storage_args(self):
        if self.args[0] == 'list' and '-l' in self.args:
            gcloud_storage_map = GcloudStorageMap(gcloud_command={'list': LIST_COMMAND_LONG_FORMAT}, flag_map={})
        else:
            gcloud_storage_map = GcloudStorageMap(gcloud_command={'create': CREATE_COMMAND, 'delete': DELETE_COMMAND, 'update': UPDATE_COMMAND, 'get': GET_COMMAND, 'list': LIST_COMMAND}, flag_map={})
        return super().get_gcloud_storage_args(gcloud_storage_map)

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

    def _DeleteHmacKey(self, thread_state=None):
        """Deletes an HMAC key."""
        if self.args:
            access_id = self.args[0]
        else:
            raise _AccessIdException(self.command_name, self.action_subcommand, _DELETE_SYNOPSIS)
        gsutil_api = GetCloudApiInstance(self, thread_state=thread_state)
        gsutil_api.DeleteHmacKey(self.project_id, access_id, provider='gs')

    def _GetHmacKey(self, thread_state=None):
        """Gets HMAC key from its Access Id."""
        if self.args:
            access_id = self.args[0]
        else:
            raise _AccessIdException(self.command_name, self.action_subcommand, _GET_SYNOPSIS)
        gsutil_api = GetCloudApiInstance(self, thread_state=thread_state)
        response = gsutil_api.GetHmacKey(self.project_id, access_id, provider='gs')
        print(_KeyMetadataOutput(response))

    def _ListHmacKeys(self, thread_state=None):
        """Lists HMAC keys for a project or service account."""
        if self.args:
            raise CommandException('%s %s received unexpected arguments.\n%s' % (self.command_name, self.action_subcommand, _LIST_SYNOPSIS))
        gsutil_api = GetCloudApiInstance(self, thread_state=thread_state)
        response = gsutil_api.ListHmacKeys(self.project_id, self.service_account_email, self.show_all, provider='gs')
        short_list_format = '%s\t%-12s %s'
        if self.long_list:
            for item in response:
                print(_KeyMetadataOutput(item))
                print()
        else:
            for item in response:
                print(short_list_format % (item.accessId, item.state, item.serviceAccountEmail))

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

    def RunCommand(self):
        """Command entry point for the hmac command."""
        if self.gsutil_api.GetApiSelector(provider='gs') != ApiSelector.JSON:
            raise CommandException('The "hmac" command can only be used with the GCS JSON API')
        self.action_subcommand = self.args.pop(0)
        self.ParseSubOpts(check_args=True)
        LogCommandParams(sub_opts=self.sub_opts)
        self.service_account_email = None
        self.state = None
        self.show_all = False
        self.long_list = False
        self.etag = None
        if self.sub_opts:
            for o, a in self.sub_opts:
                if o == '-u':
                    self.service_account_email = a
                elif o == '-p':
                    InsistAscii(a, 'Invalid non-ASCII character found in project ID')
                    self.project_id = a
                elif o == '-s':
                    self.state = a
                elif o == '-a':
                    self.show_all = True
                elif o == '-l':
                    self.long_list = True
                elif o == '-e':
                    self.etag = a
        if not self.project_id:
            self.project_id = PopulateProjectId(None)
        method_for_arg = {'create': self._CreateHmacKey, 'delete': self._DeleteHmacKey, 'get': self._GetHmacKey, 'list': self._ListHmacKeys, 'update': self._UpdateHmacKey}
        if self.action_subcommand not in method_for_arg:
            raise CommandException('Invalid subcommand "%s" for the %s command.\nSee "gsutil help hmac".' % (self.action_subcommand, self.command_name))
        LogCommandParams(subcommands=[self.action_subcommand])
        method_for_arg[self.action_subcommand]()
        return 0