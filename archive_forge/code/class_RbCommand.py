from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.cloud_api import NotEmptyException
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.storage_url import StorageUrlFromString
from gslib.utils.constants import NO_MAX
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
class RbCommand(Command):
    """Implementation of gsutil rb command."""
    command_spec = Command.CreateCommandSpec('rb', command_name_aliases=['deletebucket', 'removebucket', 'removebuckets', 'rmdir'], usage_synopsis=_SYNOPSIS, min_args=1, max_args=NO_MAX, supported_sub_args='f', file_url_ok=False, provider_url_ok=False, urls_start_arg=0, gs_api_support=[ApiSelector.XML, ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments=[CommandArgument.MakeZeroOrMoreCloudBucketURLsArgument()])
    help_spec = Command.HelpSpec(help_name='rb', help_name_aliases=['deletebucket', 'removebucket', 'removebuckets', 'rmdir'], help_type='command_help', help_one_line_summary='Remove buckets', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={})
    gcloud_storage_map = GcloudStorageMap(gcloud_command=['storage', 'buckets', 'delete'], flag_map={'-f': GcloudStorageFlag('--continue-on-error')})

    def RunCommand(self):
        """Command entry point for the rb command."""
        self.continue_on_error = False
        if self.sub_opts:
            for o, unused_a in self.sub_opts:
                if o == '-f':
                    self.continue_on_error = True
        did_some_work = False
        some_failed = False
        for url_str in self.args:
            wildcard_url = StorageUrlFromString(url_str)
            if wildcard_url.IsObject():
                raise CommandException('"rb" command requires a provider or bucket URL')
            try:
                blrs = list(self.WildcardIterator(url_str).IterBuckets(bucket_fields=['id']))
            except:
                some_failed = True
                if self.continue_on_error:
                    continue
                else:
                    raise
            for blr in blrs:
                url = blr.storage_url
                self.logger.info('Removing %s...', url)
                try:
                    self.gsutil_api.DeleteBucket(url.bucket_name, provider=url.scheme)
                except NotEmptyException as e:
                    some_failed = True
                    if self.continue_on_error:
                        continue
                    elif 'VersionedBucketNotEmpty' in e.reason:
                        raise CommandException('Bucket is not empty. Note: this is a versioned bucket, so to delete all objects\nyou need to use:\n\tgsutil rm -r %s' % url)
                    else:
                        raise
                except:
                    some_failed = True
                    if not self.continue_on_error:
                        raise
                did_some_work = True
        if not did_some_work:
            raise CommandException(NO_URLS_MATCHED_TARGET % list(self.args))
        return 1 if some_failed else 0