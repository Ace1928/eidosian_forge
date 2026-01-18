from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.commands.cp import CP_AND_MV_SHIM_FLAG_MAP
from gslib.commands.cp import CP_SUB_ARGS
from gslib.commands.cp import ShimTranslatePredefinedAclSubOptForCopy
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.storage_url import StorageUrlFromString
from gslib.utils.constants import NO_MAX
from gslib.utils.shim_util import GcloudStorageMap
class MvCommand(Command):
    """Implementation of gsutil mv command.

     Note that there is no atomic rename operation - this command is simply
     a shorthand for 'cp' followed by 'rm'.
  """
    command_spec = Command.CreateCommandSpec('mv', command_name_aliases=['move', 'ren', 'rename'], usage_synopsis=_SYNOPSIS, min_args=1, max_args=NO_MAX, supported_sub_args=CP_SUB_ARGS, file_url_ok=True, provider_url_ok=False, urls_start_arg=0, gs_api_support=[ApiSelector.XML, ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments=[CommandArgument.MakeZeroOrMoreCloudOrFileURLsArgument()])
    help_spec = Command.HelpSpec(help_name='mv', help_name_aliases=['move', 'rename'], help_type='command_help', help_one_line_summary='Move/rename objects', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={})

    def get_gcloud_storage_args(self):
        ShimTranslatePredefinedAclSubOptForCopy(self.sub_opts)
        gcloud_storage_map = GcloudStorageMap(gcloud_command=['storage', 'mv'], flag_map=CP_AND_MV_SHIM_FLAG_MAP)
        return super().get_gcloud_storage_args(gcloud_storage_map)

    def RunCommand(self):
        """Command entry point for the mv command."""
        for arg_to_check in self.args[0:-1]:
            url = StorageUrlFromString(arg_to_check)
            if url.IsCloudUrl() and (url.IsBucket() or url.IsProvider()):
                raise CommandException('You cannot move a source bucket using the mv command. If you meant to move\nall objects in the bucket, you can use a command like:\n\tgsutil mv %s/* %s' % (arg_to_check, self.args[-1]))
        unparsed_args = ['-M']
        if self.recursion_requested:
            unparsed_args.append('-R')
        unparsed_args.extend(self.unparsed_args)
        self.command_runner.RunNamedCommand('cp', args=unparsed_args, headers=self.headers, debug=self.debug, trace_token=self.trace_token, user_project=self.user_project, parallel_operations=self.parallel_operations)
        return 0