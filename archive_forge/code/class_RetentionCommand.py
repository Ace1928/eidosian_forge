from __future__ import absolute_import
import time
from apitools.base.py import encoding
from gslib import metrics
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import Preconditions
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.help_provider import CreateHelpText
from gslib.name_expansion import NameExpansionIterator
from gslib.name_expansion import SeekAheadNameExpansionIterator
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.thread_message import MetadataMessage
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.constants import NO_MAX
from gslib.utils.parallelism_framework_util import PutToQueueWithTimeout
from gslib.utils.retention_util import ConfirmLockRequest
from gslib.utils.retention_util import ReleaseEventHoldFuncWrapper
from gslib.utils.retention_util import ReleaseTempHoldFuncWrapper
from gslib.utils.retention_util import RetentionInSeconds
from gslib.utils.retention_util import RetentionPolicyToString
from gslib.utils.retention_util import SetEventHoldFuncWrapper
from gslib.utils.retention_util import SetTempHoldFuncWrapper
from gslib.utils.retention_util import UpdateObjectMetadataExceptionHandler
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.translation_helper import PreconditionsFromHeaders
class RetentionCommand(Command):
    """Implementation of gsutil retention command."""
    command_spec = Command.CreateCommandSpec('retention', command_name_aliases=[], usage_synopsis=_SYNOPSIS, min_args=2, max_args=NO_MAX, file_url_ok=False, provider_url_ok=False, urls_start_arg=1, gs_api_support=[ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments={'set': [CommandArgument.MakeZeroOrMoreCloudBucketURLsArgument()], 'clear': [CommandArgument.MakeZeroOrMoreCloudBucketURLsArgument()], 'get': [CommandArgument.MakeNCloudBucketURLsArgument(1)], 'lock': [CommandArgument.MakeZeroOrMoreCloudBucketURLsArgument()], 'event-default': {'set': [CommandArgument.MakeZeroOrMoreCloudURLsArgument()], 'release': [CommandArgument.MakeZeroOrMoreCloudURLsArgument()]}, 'event': {'set': [CommandArgument.MakeZeroOrMoreCloudURLsArgument()], 'release': [CommandArgument.MakeZeroOrMoreCloudURLsArgument()]}, 'temp': {'set': [CommandArgument.MakeZeroOrMoreCloudURLsArgument()], 'release': [CommandArgument.MakeZeroOrMoreCloudURLsArgument()]}})
    help_spec = Command.HelpSpec(help_name='retention', help_name_aliases=[], help_type='command_help', help_one_line_summary='Provides utilities to interact with Retention Policy feature.', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={'get': _get_help_text, 'set': _set_help_text, 'clear': _clear_help_text, 'lock': _lock_help_text, 'event-default': _event_default_help_text, 'event': _event_help_text, 'temp': _temp_help_text})

    def get_gcloud_storage_args(self):
        if self.args[0] == 'set':
            gcloud_storage_map = GcloudStorageMap(gcloud_command={'set': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'update', '--retention-period={}s'.format(RetentionInSeconds(self.args[1]))] + self.args[2:], flag_map={})}, flag_map={})
            self.args = self.args[:1]
        else:
            gcloud_storage_map = GcloudStorageMap(gcloud_command={'clear': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'update', '--clear-retention-period'], flag_map={}), 'event': GcloudStorageMap(gcloud_command={'set': GcloudStorageMap(gcloud_command=['storage', 'objects', 'update', '--event-based-hold'], flag_map={}), 'release': GcloudStorageMap(gcloud_command=['storage', 'objects', 'update', '--no-event-based-hold'], flag_map={})}, flag_map={}), 'event-default': GcloudStorageMap(gcloud_command={'set': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'update', '--default-event-based-hold'], flag_map={}), 'release': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'update', '--no-default-event-based-hold'], flag_map={})}, flag_map={}), 'get': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'describe', '--format=yaml(retentionPolicy)', '--raw'], flag_map={}), 'lock': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'update', '--lock-retention-period'], flag_map={}), 'temp': GcloudStorageMap(gcloud_command={'set': GcloudStorageMap(gcloud_command=['storage', 'objects', 'update', '--temporary-hold'], flag_map={}), 'release': GcloudStorageMap(gcloud_command=['storage', 'objects', 'update', '--no-temporary-hold'], flag_map={})}, flag_map={})}, flag_map={})
        return super().get_gcloud_storage_args(gcloud_storage_map)

    def RunCommand(self):
        """Command entry point for the retention command."""
        if self.gsutil_api.GetApiSelector('gs') != ApiSelector.JSON:
            raise CommandException('The {} command can only be used with the GCS JSON API. If you have only supplied hmac credentials in your boto file, please instead supply a credential type that can be used with the JSON API.'.format(self.command_name))
        self.preconditions = PreconditionsFromHeaders(self.headers)
        action_subcommand = self.args.pop(0)
        self.ParseSubOpts(check_args=True)
        if action_subcommand == 'set':
            func = self._SetRetention
        elif action_subcommand == 'clear':
            func = self._ClearRetention
        elif action_subcommand == 'get':
            func = self._GetRetention
        elif action_subcommand == 'lock':
            func = self._LockRetention
        elif action_subcommand == 'event-default':
            func = self._DefaultEventHold
        elif action_subcommand == 'event':
            func = self._EventHold
        elif action_subcommand == 'temp':
            func = self._TempHold
        else:
            raise CommandException('Invalid subcommand "{}" for the {} command.\nSee "gsutil help retention".'.format(action_subcommand, self.command_name))
        metrics.LogCommandParams(subcommands=[action_subcommand], sub_opts=self.sub_opts)
        return func()

    def BucketUpdateFunc(self, url_args, bucket_metadata_update, fields, log_msg_template):
        preconditions = Preconditions(meta_gen_match=self.preconditions.meta_gen_match)
        some_matched = False
        for url_str in url_args:
            bucket_iter = self.GetBucketUrlIterFromArg(url_str, bucket_fields=['id'])
            for blr in bucket_iter:
                url = blr.storage_url
                some_matched = True
                self.logger.info(log_msg_template, blr)
                self.gsutil_api.PatchBucket(url.bucket_name, bucket_metadata_update, preconditions=preconditions, provider=url.scheme, fields=fields)
        if not some_matched:
            raise CommandException(NO_URLS_MATCHED_TARGET % list(url_args))

    def ObjectUpdateMetadataFunc(self, patch_obj_metadata, log_template, name_expansion_result, thread_state=None):
        """Updates metadata on an object using PatchObjectMetadata.

    Args:
      patch_obj_metadata: Metadata changes that should be applied to the
                          existing object.
      log_template: The log template that should be printed for each object.
      name_expansion_result: NameExpansionResult describing target object.
      thread_state: gsutil Cloud API instance to use for the operation.
    """
        gsutil_api = GetCloudApiInstance(self, thread_state=thread_state)
        exp_src_url = name_expansion_result.expanded_storage_url
        self.logger.info(log_template, exp_src_url)
        cloud_obj_metadata = encoding.JsonToMessage(apitools_messages.Object, name_expansion_result.expanded_result)
        preconditions = Preconditions(gen_match=self.preconditions.gen_match, meta_gen_match=self.preconditions.meta_gen_match)
        if preconditions.gen_match is None:
            preconditions.gen_match = cloud_obj_metadata.generation
        if preconditions.meta_gen_match is None:
            preconditions.meta_gen_match = cloud_obj_metadata.metageneration
        gsutil_api.PatchObjectMetadata(exp_src_url.bucket_name, exp_src_url.object_name, patch_obj_metadata, generation=exp_src_url.generation, preconditions=preconditions, provider=exp_src_url.scheme, fields=['id'])
        PutToQueueWithTimeout(gsutil_api.status_queue, MetadataMessage(message_time=time.time()))

    def _GetObjectNameExpansionIterator(self, url_args):
        return NameExpansionIterator(self.command_name, self.debug, self.logger, self.gsutil_api, url_args, self.recursion_requested, all_versions=self.all_versions, continue_on_error=self.parallel_operations, bucket_listing_fields=['generation', 'metageneration'])

    def _GetSeekAheadNameExpansionIterator(self, url_args):
        return SeekAheadNameExpansionIterator(self.command_name, self.debug, self.GetSeekAheadGsutilApi(), url_args, self.recursion_requested, all_versions=self.all_versions, project_id=self.project_id)

    def _SetRetention(self):
        """Set retention retention_period on one or more buckets."""
        seconds = RetentionInSeconds(self.args[0])
        retention_policy = apitools_messages.Bucket.RetentionPolicyValue(retentionPeriod=seconds)
        log_msg_template = 'Setting Retention Policy on %s...'
        bucket_metadata_update = apitools_messages.Bucket(retentionPolicy=retention_policy)
        url_args = self.args[1:]
        self.BucketUpdateFunc(url_args, bucket_metadata_update, fields=['id', 'retentionPolicy'], log_msg_template=log_msg_template)
        return 0

    def _ClearRetention(self):
        """Clear retention retention_period on one or more buckets."""
        retention_policy = apitools_messages.Bucket.RetentionPolicyValue(retentionPeriod=None)
        log_msg_template = 'Clearing Retention Policy on %s...'
        bucket_metadata_update = apitools_messages.Bucket(retentionPolicy=retention_policy)
        url_args = self.args
        self.BucketUpdateFunc(url_args, bucket_metadata_update, fields=['id', 'retentionPolicy'], log_msg_template=log_msg_template)
        return 0

    def _GetRetention(self):
        """Get Retention Policy for a single bucket."""
        bucket_url, bucket_metadata = self.GetSingleBucketUrlFromArg(self.args[0], bucket_fields=['retentionPolicy'])
        print(RetentionPolicyToString(bucket_metadata.retentionPolicy, bucket_url))
        return 0

    def _LockRetention(self):
        """Lock Retention Policy on one or more buckets."""
        url_args = self.args
        some_matched = False
        for url_str in url_args:
            bucket_iter = self.GetBucketUrlIterFromArg(url_str, bucket_fields=['id'])
            for blr in bucket_iter:
                url = blr.storage_url
                some_matched = True
                bucket_metadata = self.gsutil_api.GetBucket(url.bucket_name, provider=url.scheme, fields=['id', 'metageneration', 'retentionPolicy'])
                if not (bucket_metadata.retentionPolicy and bucket_metadata.retentionPolicy.retentionPeriod):
                    raise CommandException('Bucket "{}" does not have an Unlocked Retention Policy.'.format(url.bucket_name))
                elif bucket_metadata.retentionPolicy.isLocked is True:
                    self.logger.error('Retention Policy on "%s" is already locked.', blr)
                elif ConfirmLockRequest(url.bucket_name, bucket_metadata.retentionPolicy):
                    self.logger.info('Locking Retention Policy on %s...', blr)
                    self.gsutil_api.LockRetentionPolicy(url.bucket_name, bucket_metadata.metageneration, provider=url.scheme)
                else:
                    self.logger.error('  Abort Locking Retention Policy on {}'.format(blr))
        if not some_matched:
            raise CommandException(NO_URLS_MATCHED_TARGET % list(url_args))
        return 0

    def _DefaultEventHold(self):
        """Sets default value for Event-Based Hold on one or more buckets."""
        hold = None
        if self.args:
            if self.args[0].lower() == 'set':
                hold = True
            elif self.args[0].lower() == 'release':
                hold = False
            else:
                raise CommandException('Invalid subcommand "{}" for the "retention event-default" command.\nSee "gsutil help retention event".'.format(self.sub_opts))
        verb = 'Setting' if hold else 'Releasing'
        log_msg_template = '{} default Event-Based Hold on %s...'.format(verb)
        bucket_metadata_update = apitools_messages.Bucket(defaultEventBasedHold=hold)
        url_args = self.args[1:]
        self.BucketUpdateFunc(url_args, bucket_metadata_update, fields=['id', 'defaultEventBasedHold'], log_msg_template=log_msg_template)
        return 0

    def _EventHold(self):
        """Sets or unsets Event-Based Hold on one or more objects."""
        sub_command_name = 'event'
        sub_command_full_name = 'Event-Based'
        hold = self._ProcessHoldArgs(sub_command_name)
        url_args = self.args[1:]
        obj_metadata_update_wrapper = SetEventHoldFuncWrapper if hold else ReleaseEventHoldFuncWrapper
        self._SetHold(obj_metadata_update_wrapper, url_args, sub_command_full_name)
        return 0

    def _TempHold(self):
        """Sets or unsets Temporary Hold on one or more objects."""
        sub_command_name = 'temp'
        sub_command_full_name = 'Temporary'
        hold = self._ProcessHoldArgs(sub_command_name)
        url_args = self.args[1:]
        obj_metadata_update_wrapper = SetTempHoldFuncWrapper if hold else ReleaseTempHoldFuncWrapper
        self._SetHold(obj_metadata_update_wrapper, url_args, sub_command_full_name)
        return 0

    def _ProcessHoldArgs(self, sub_command_name):
        """Processes command args for Temporary and Event-Based Hold sub-command.

    Args:
      sub_command_name: The name of the subcommand: "temp" / "event"

    Returns:
      Returns a boolean value indicating whether to set (True) or
      release (False)the Hold.
    """
        hold = None
        if self.args[0].lower() == 'set':
            hold = True
        elif self.args[0].lower() == 'release':
            hold = False
        else:
            raise CommandException('Invalid subcommand "{}" for the "retention {}" command.\nSee "gsutil help retention {}".'.format(self.args[0], sub_command_name, sub_command_name))
        return hold

    def _SetHold(self, obj_metadata_update_wrapper, url_args, sub_command_full_name):
        """Common logic to set or unset Event-Based/Temporary Hold on objects.

    Args:
      obj_metadata_update_wrapper: The function for updating related fields in
                                   Object metadata.
      url_args: List of object URIs.
      sub_command_full_name: The full name for sub-command:
                             "Temporary" / "Event-Based"
    """
        if len(url_args) == 1 and (not self.recursion_requested):
            url = StorageUrlFromString(url_args[0])
            if not (url.IsCloudUrl() and url.IsObject()):
                raise CommandException('URL ({}) must name an object'.format(url_args[0]))
        name_expansion_iterator = self._GetObjectNameExpansionIterator(url_args)
        seek_ahead_iterator = self._GetSeekAheadNameExpansionIterator(url_args)
        self.everything_set_okay = True
        try:
            self.Apply(obj_metadata_update_wrapper, name_expansion_iterator, UpdateObjectMetadataExceptionHandler, fail_on_error=True, seek_ahead_iterator=seek_ahead_iterator)
        except AccessDeniedException as e:
            if e.status == 403:
                self._WarnServiceAccounts()
            raise
        if not self.everything_set_okay:
            raise CommandException('{} Hold for some objects could not be set.'.format(sub_command_full_name))