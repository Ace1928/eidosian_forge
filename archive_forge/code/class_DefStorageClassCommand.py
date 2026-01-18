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
from gslib.utils.text_util import NormalizeStorageClass
from gslib.utils import shim_util
class DefStorageClassCommand(Command):
    """Implementation of gsutil defstorageclass command."""
    command_spec = Command.CreateCommandSpec('defstorageclass', usage_synopsis=_SYNOPSIS, min_args=2, max_args=NO_MAX, supported_sub_args='', file_url_ok=False, provider_url_ok=False, urls_start_arg=2, gs_api_support=[ApiSelector.XML, ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments={'set': [CommandArgument.MakeFreeTextArgument(), CommandArgument.MakeZeroOrMoreCloudBucketURLsArgument()], 'get': [CommandArgument.MakeZeroOrMoreCloudBucketURLsArgument()]})
    help_spec = Command.HelpSpec(help_name='defstorageclass', help_name_aliases=['defaultstorageclass'], help_type='command_help', help_one_line_summary='Get or set the default storage class on buckets', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={'get': _get_help_text, 'set': _set_help_text})
    gcloud_storage_map = GcloudStorageMap(gcloud_command={'get': SHIM_GET_COMMAND_MAP, 'set': SHIM_SET_COMMAND_MAP}, flag_map={})

    def _CheckIsGsUrl(self, url_str):
        if not url_str.startswith('gs://'):
            raise CommandException('"%s" does not support the URL "%s". Did you mean to use a gs:// URL?' % (self.command_name, url_str))

    def _CalculateUrlsStartArg(self):
        if not self.args:
            self.RaiseWrongNumberOfArgumentsException()
        if self.args[0].lower() == 'set':
            return 2
        else:
            return 1

    def _SetDefStorageClass(self):
        """Sets the default storage class for a bucket."""
        normalized_storage_class = NormalizeStorageClass(self.args[0])
        url_args = self.args[1:]
        if not url_args:
            self.RaiseWrongNumberOfArgumentsException()
        some_matched = False
        for url_str in url_args:
            self._CheckIsGsUrl(url_str)
            bucket_iter = self.GetBucketUrlIterFromArg(url_str, bucket_fields=['id'])
            for blr in bucket_iter:
                some_matched = True
                bucket_metadata = apitools_messages.Bucket()
                self.logger.info('Setting default storage class to "%s" for bucket %s' % (normalized_storage_class, blr.url_string.rstrip('/')))
                bucket_metadata.storageClass = normalized_storage_class
                self.gsutil_api.PatchBucket(blr.storage_url.bucket_name, bucket_metadata, provider=blr.storage_url.scheme, fields=['id'])
        if not some_matched:
            raise CommandException(NO_URLS_MATCHED_TARGET % list(url_args))

    def _GetDefStorageClass(self):
        """Gets the default storage class for a bucket."""
        url_args = self.args
        some_matched = False
        for url_str in url_args:
            self._CheckIsGsUrl(url_str)
            bucket_iter = self.GetBucketUrlIterFromArg(url_str, bucket_fields=['storageClass'])
            for blr in bucket_iter:
                some_matched = True
                print('%s: %s' % (blr.url_string.rstrip('/'), blr.root_object.storageClass))
        if not some_matched:
            raise CommandException(NO_URLS_MATCHED_TARGET % list(url_args))

    def RunCommand(self):
        """Command entry point for the defstorageclass command."""
        action_subcommand = self.args.pop(0)
        subcommand_args = [action_subcommand]
        if action_subcommand == 'get':
            func = self._GetDefStorageClass
        elif action_subcommand == 'set':
            func = self._SetDefStorageClass
            normalized_storage_class = NormalizeStorageClass(self.args[0])
            subcommand_args.append(normalized_storage_class)
        else:
            raise CommandException('Invalid subcommand "%s" for the %s command.\nSee "gsutil help %s".' % (action_subcommand, self.command_name, self.command_name))
        metrics.LogCommandParams(subcommands=subcommand_args)
        func()
        return 0