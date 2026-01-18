from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib import metrics
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.help_provider import CreateHelpText
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import NO_MAX
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils import shim_util
class RequesterPaysCommand(Command):
    """Implementation of gsutil requesterpays command."""
    command_spec = Command.CreateCommandSpec('requesterpays', usage_synopsis=_SYNOPSIS, min_args=2, max_args=NO_MAX, supported_sub_args='', file_url_ok=False, provider_url_ok=False, urls_start_arg=2, gs_api_support=[ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments={'set': [CommandArgument('mode', choices=['on', 'off']), CommandArgument.MakeZeroOrMoreCloudBucketURLsArgument()], 'get': [CommandArgument.MakeZeroOrMoreCloudBucketURLsArgument()]})
    help_spec = Command.HelpSpec(help_name='requesterpays', help_name_aliases=[], help_type='command_help', help_one_line_summary='Enable or disable requester pays for one or more buckets', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={'get': _get_help_text, 'set': _set_help_text})
    gcloud_storage_map = GcloudStorageMap(gcloud_command={'get': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'list', _GCLOUD_FORMAT_STRING], flag_map={}, supports_output_translation=True), 'set': GcloudStorageMap(gcloud_command={'on': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'update', '--requester-pays'], flag_map={}), 'off': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'update', '--no-requester-pays'], flag_map={})}, flag_map={})}, flag_map={})

    def _CalculateUrlsStartArg(self):
        if not self.args:
            self.RaiseWrongNumberOfArgumentsException()
        if self.args[0].lower() == 'set':
            return 2
        else:
            return 1

    def _SetRequesterPays(self):
        """Gets requesterpays configuration for a bucket."""
        requesterpays_arg = self.args[0].lower()
        if requesterpays_arg not in ('on', 'off'):
            raise CommandException('Argument to "%s set" must be either <on|off>' % self.command_name)
        url_args = self.args[1:]
        if not url_args:
            self.RaiseWrongNumberOfArgumentsException()
        some_matched = False
        for url_str in url_args:
            bucket_iter = self.GetBucketUrlIterFromArg(url_str, bucket_fields=['id'])
            for blr in bucket_iter:
                url = blr.storage_url
                some_matched = True
                bucket_metadata = apitools_messages.Bucket(billing=apitools_messages.Bucket.BillingValue())
                if requesterpays_arg == 'on':
                    self.logger.info('Enabling requester pays for %s...', url)
                    bucket_metadata.billing.requesterPays = True
                else:
                    self.logger.info('Disabling requester pays for %s...', url)
                    bucket_metadata.billing.requesterPays = False
                self.gsutil_api.PatchBucket(url.bucket_name, bucket_metadata, provider=url.scheme, fields=['id'])
        if not some_matched:
            raise CommandException(NO_URLS_MATCHED_TARGET % list(url_args))

    def _GetRequesterPays(self):
        """Gets requesterpays configuration for one or more buckets."""
        url_args = self.args
        some_matched = False
        for url_str in url_args:
            bucket_iter = self.GetBucketUrlIterFromArg(url_str, bucket_fields=['billing'])
            for blr in bucket_iter:
                some_matched = True
                if blr.root_object.billing and blr.root_object.billing.requesterPays:
                    print('%s: Enabled' % blr.url_string.rstrip('/'))
                else:
                    print('%s: Disabled' % blr.url_string.rstrip('/'))
        if not some_matched:
            raise CommandException(NO_URLS_MATCHED_TARGET % list(url_args))

    def RunCommand(self):
        """Command entry point for the requesterpays command."""
        action_subcommand = self.args.pop(0)
        if action_subcommand == 'get':
            func = self._GetRequesterPays
            metrics.LogCommandParams(subcommands=[action_subcommand])
        elif action_subcommand == 'set':
            func = self._SetRequesterPays
            requesterpays_arg = self.args[0].lower()
            if requesterpays_arg in ('on', 'off'):
                metrics.LogCommandParams(subcommands=[action_subcommand, requesterpays_arg])
        else:
            raise CommandException('Invalid subcommand "%s" for the %s command.\nSee "gsutil help %s".' % (action_subcommand, self.command_name, self.command_name))
        func()
        return 0