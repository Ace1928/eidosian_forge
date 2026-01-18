from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import difflib
import logging
import os
import pkgutil
import sys
import textwrap
import time
import six
from six.moves import input
import boto
from boto import config
from boto.storage_uri import BucketStorageUri
import gslib
from gslib import metrics
from gslib.cloud_api_delegator import CloudApiDelegator
from gslib.command import Command
from gslib.command import CreateOrGetGsutilLogger
from gslib.command import GetFailureCount
from gslib.command import OLD_ALIAS_MAP
from gslib.command import ShutDownGsutil
import gslib.commands
from gslib.cs_api_map import ApiSelector
from gslib.cs_api_map import GsutilApiClassMapFactory
from gslib.cs_api_map import GsutilApiMapFactory
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.exception import CommandException
from gslib.gcs_json_api import GcsJsonApi
from gslib.no_op_credentials import NoOpCredentials
from gslib.tab_complete import MakeCompleter
from gslib.utils import boto_util
from gslib.utils import shim_util
from gslib.utils import system_util
from gslib.utils.constants import RELEASE_NOTES_URL
from gslib.utils.constants import UTF8
from gslib.utils.metadata_util import IsCustomMetadataHeader
from gslib.utils.parallelism_framework_util import CheckMultiprocessingAvailableAndInit
from gslib.utils.text_util import CompareVersions
from gslib.utils.text_util import InsistAsciiHeader
from gslib.utils.text_util import InsistAsciiHeaderValue
from gslib.utils.text_util import print_to_fd
from gslib.utils.unit_util import SECONDS_PER_DAY
from gslib.utils.update_util import LookUpGsutilVersion
from gslib.utils.update_util import GsutilPubTarball
def RunNamedCommand(self, command_name, args=None, headers=None, debug=0, trace_token=None, parallel_operations=False, skip_update_check=False, logging_filters=None, do_shutdown=True, perf_trace_token=None, user_project=None, collect_analytics=False):
    """Runs the named command.

    Used by gsutil main, commands built atop other commands, and tests.

    Args:
      command_name: The name of the command being run.
      args: Command-line args (arg0 = actual arg, not command name ala bash).
      headers: Dictionary containing optional HTTP headers to pass to boto.
      debug: Debug level to pass in to boto connection (range 0..3).
      trace_token: Trace token to pass to the underlying API.
      parallel_operations: Should command operations be executed in parallel?
      skip_update_check: Set to True to disable checking for gsutil updates.
      logging_filters: Optional list of logging.Filters to apply to this
          command's logger.
      do_shutdown: Stop all parallelism framework workers iff this is True.
      perf_trace_token: Performance measurement trace token to pass to the
          underlying API.
      user_project: The project to bill this request to.
      collect_analytics: Set to True to collect an analytics metric logging this
          command.

    Raises:
      CommandException: if errors encountered.

    Returns:
      Return value(s) from Command that was run.
    """
    command_changed_to_update = False
    if not skip_update_check and self.MaybeCheckForAndOfferSoftwareUpdate(command_name, debug):
        command_name = 'update'
        command_changed_to_update = True
        args = [_StringToSysArgType('-n')]
        if system_util.IsRunningInteractively() and collect_analytics:
            metrics.CheckAndMaybePromptForAnalyticsEnabling()
    self.MaybePromptForPythonUpdate(command_name)
    if not args:
        args = []
    api_version = boto.config.get_value('GSUtil', 'default_api_version', '1')
    if not headers:
        headers = {}
    headers['x-goog-api-version'] = api_version
    if command_name not in self.command_map:
        close_matches = difflib.get_close_matches(command_name, self.command_map.keys(), n=1)
        if close_matches:
            translated_command_name = OLD_ALIAS_MAP.get(close_matches[0], close_matches)[0]
            print('Did you mean this?', file=sys.stderr)
            print('\t%s' % translated_command_name, file=sys.stderr)
        elif command_name == 'update' and gslib.IS_PACKAGE_INSTALL:
            sys.stderr.write('Update command is not supported for package installs; please instead update using your package manager.')
        raise CommandException('Invalid command "%s".' % command_name)
    if _StringToSysArgType('--help') in args:
        new_args = [command_name]
        original_command_class = self.command_map[command_name]
        subcommands = original_command_class.help_spec.subcommand_help_text.keys()
        for arg in args:
            if arg in subcommands:
                new_args.append(arg)
                break
        args = new_args
        command_name = 'help'
    HandleArgCoding(args)
    HandleHeaderCoding(headers)
    command_class = self.command_map[command_name]
    command_inst = command_class(self, args, headers, debug, trace_token, parallel_operations, self.bucket_storage_uri_class, self.gsutil_api_class_map_factory, logging_filters, command_alias_used=command_name, perf_trace_token=perf_trace_token, user_project=user_project)
    if collect_analytics:
        metrics.LogCommandParams(command_name=command_inst.command_name, sub_opts=command_inst.sub_opts, command_alias=command_name)
    if command_inst.translate_to_gcloud_storage_if_requested():
        return_code = command_inst.run_gcloud_storage()
    else:
        return_code = command_inst.RunCommand()
    if CheckMultiprocessingAvailableAndInit().is_available and do_shutdown:
        ShutDownGsutil()
    if GetFailureCount() > 0:
        return_code = 1
    if command_changed_to_update:
        return_code = 1
        print('\n'.join(textwrap.wrap('Update was successful. Exiting with code 1 as the original command issued prior to the update was not executed and should be re-run.')))
    return return_code