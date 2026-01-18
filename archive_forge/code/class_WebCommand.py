from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import sys
from apitools.base.py import encoding
from gslib import metrics
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.help_provider import CreateHelpText
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import NO_MAX
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
class WebCommand(Command):
    """Implementation of gsutil web command."""
    command_spec = Command.CreateCommandSpec('web', command_name_aliases=['setwebcfg', 'getwebcfg'], usage_synopsis=_SYNOPSIS, min_args=2, max_args=NO_MAX, supported_sub_args='m:e:', file_url_ok=False, provider_url_ok=False, urls_start_arg=1, gs_api_support=[ApiSelector.XML, ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments={'set': [CommandArgument.MakeZeroOrMoreCloudBucketURLsArgument()], 'get': [CommandArgument.MakeNCloudBucketURLsArgument(1)]})
    help_spec = Command.HelpSpec(help_name='web', help_name_aliases=['getwebcfg', 'setwebcfg'], help_type='command_help', help_one_line_summary='Set a website configuration for a bucket', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={'get': _get_help_text, 'set': _set_help_text})
    gcloud_storage_map = GcloudStorageMap(gcloud_command={'get': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'describe', '--format="gsutiljson[key=website_config,empty=\' has no website configuration.\',empty_prefix_key=storage_url]"', '--raw'], flag_map={}, supports_output_translation=True)}, flag_map={})

    def get_gcloud_storage_args(self):
        if self.args[0] == 'set':
            set_command_map = GcloudStorageMap(gcloud_command={'set': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'update'], flag_map={'-e': GcloudStorageFlag('--web-error-page'), '-m': GcloudStorageFlag('--web-main-page-suffix')})}, flag_map={})
            if not ('-e' in self.args or '-m' in self.args):
                set_command_map.gcloud_command['set'].gcloud_command += ['--clear-web-error-page', '--clear-web-main-page-suffix']
            gcloud_storage_map = set_command_map
        else:
            gcloud_storage_map = WebCommand.gcloud_storage_map
        return super().get_gcloud_storage_args(gcloud_storage_map)

    def _GetWeb(self):
        """Gets website configuration for a bucket."""
        bucket_url, bucket_metadata = self.GetSingleBucketUrlFromArg(self.args[0], bucket_fields=['website'])
        if bucket_url.scheme == 's3':
            sys.stdout.write(self.gsutil_api.XmlPassThroughGetWebsite(bucket_url, provider=bucket_url.scheme))
        elif bucket_metadata.website and (bucket_metadata.website.mainPageSuffix or bucket_metadata.website.notFoundPage):
            sys.stdout.write(str(encoding.MessageToJson(bucket_metadata.website)) + '\n')
        else:
            sys.stdout.write('%s has no website configuration.\n' % bucket_url)
        return 0

    def _SetWeb(self):
        """Sets website configuration for a bucket."""
        main_page_suffix = None
        error_page = None
        if self.sub_opts:
            for o, a in self.sub_opts:
                if o == '-m':
                    main_page_suffix = a
                elif o == '-e':
                    error_page = a
        url_args = self.args
        website = apitools_messages.Bucket.WebsiteValue(mainPageSuffix=main_page_suffix, notFoundPage=error_page)
        some_matched = False
        for url_str in url_args:
            bucket_iter = self.GetBucketUrlIterFromArg(url_str, bucket_fields=['id'])
            for blr in bucket_iter:
                url = blr.storage_url
                some_matched = True
                self.logger.info('Setting website configuration on %s...', blr)
                bucket_metadata = apitools_messages.Bucket(website=website)
                self.gsutil_api.PatchBucket(url.bucket_name, bucket_metadata, provider=url.scheme, fields=['id'])
        if not some_matched:
            raise CommandException(NO_URLS_MATCHED_TARGET % list(url_args))
        return 0

    def RunCommand(self):
        """Command entry point for the web command."""
        action_subcommand = self.args.pop(0)
        self.ParseSubOpts(check_args=True)
        if action_subcommand == 'get':
            func = self._GetWeb
        elif action_subcommand == 'set':
            func = self._SetWeb
        else:
            raise CommandException('Invalid subcommand "%s" for the %s command.\nSee "gsutil help web".' % (action_subcommand, self.command_name))
        metrics.LogCommandParams(subcommands=[action_subcommand], sub_opts=self.sub_opts)
        return func()