from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import itertools
import json
import os
import re
import subprocess
import textwrap
import six
from six.moves import zip
from apitools.base.protorpclite import protojson
from apitools.base.protorpclite.messages import DecodeError
from boto import config
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import ServiceException
from gslib.command import Command
from gslib.command import GetFailureCount
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import IamChOnResourceWithConditionsException
from gslib.help_provider import CreateHelpText
from gslib.metrics import LogCommandParams
from gslib.name_expansion import NameExpansionIterator
from gslib.name_expansion import SeekAheadNameExpansionIterator
from gslib.plurality_checkable_iterator import PluralityCheckableIterator
from gslib.storage_url import GetSchemeFromUrlString
from gslib.storage_url import IsKnownUrlScheme
from gslib.storage_url import StorageUrlFromString
from gslib.storage_url import UrlsAreMixOfBucketsAndObjects
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import shim_util
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.constants import IAM_POLICY_VERSION
from gslib.utils.constants import NO_MAX
from gslib.utils import iam_helper
from gslib.utils.iam_helper import BindingStringToTuple
from gslib.utils.iam_helper import BindingsTuple
from gslib.utils.iam_helper import DeserializeBindingsTuple
from gslib.utils.iam_helper import IsEqualBindings
from gslib.utils.iam_helper import PatchBindings
from gslib.utils.iam_helper import SerializeBindingsTuple
from gslib.utils.retry_util import Retry
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.shim_util import GcloudStorageFlag
class IamCommand(Command):
    """Implementation of gsutil iam command."""
    command_spec = Command.CreateCommandSpec('iam', usage_synopsis=_SYNOPSIS, min_args=2, max_args=NO_MAX, supported_sub_args='afRrd:e:', file_url_ok=True, provider_url_ok=False, urls_start_arg=1, gs_api_support=[ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments={'get': [CommandArgument.MakeNCloudURLsArgument(1)], 'set': [CommandArgument.MakeNFileURLsArgument(1), CommandArgument.MakeZeroOrMoreCloudURLsArgument()], 'ch': [CommandArgument.MakeOneOrMoreBindingsArgument(), CommandArgument.MakeZeroOrMoreCloudURLsArgument()]})
    help_spec = Command.HelpSpec(help_name='iam', help_name_aliases=[], help_type='command_help', help_one_line_summary='Get, set, or change bucket and/or object IAM permissions.', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={'get': _get_help_text, 'set': _set_help_text, 'ch': _ch_help_text})

    def _get_resource_type(self, url_patterns):
        if self.recursion_requested or url_patterns[0].IsObject():
            return 'objects'
        else:
            return 'buckets'

    def get_gcloud_storage_args(self):
        self.sub_command = self.args.pop(0)
        if self.sub_command == 'get':
            if StorageUrlFromString(self.args[0]).IsObject():
                command_group = 'objects'
            else:
                command_group = 'buckets'
            gcloud_storage_map = GcloudStorageMap(gcloud_command=['storage', command_group, 'get-iam-policy', '--format=json'], flag_map={})
        elif self.sub_command == 'set':
            gcloud_storage_map = GcloudStorageMap(gcloud_command=['storage', None, 'set-iam-policy', '--format=json'], flag_map={'-a': GcloudStorageFlag('--all-versions'), '-e': GcloudStorageFlag('--etag'), '_empty_etag': GcloudStorageFlag('--etag='), '-f': GcloudStorageFlag('--continue-on-error'), '-R': GcloudStorageFlag('--recursive'), '-r': GcloudStorageFlag('--recursive')})
            self.ParseSubOpts()
            url_strings = self.args[1:]
            url_objects = list(map(StorageUrlFromString, url_strings))
            recurse = False
            for i, (flag_key, flag_value) in enumerate(self.sub_opts):
                if flag_key in ('-r', '-R'):
                    recurse = True
                elif flag_key == '-e' and flag_value == '':
                    self.sub_opts[i] = ('_empty_etag', '')
            _RaiseErrorIfUrlsAreMixOfBucketsAndObjects(url_objects, recurse)
            if recurse or url_objects[0].IsObject():
                gcloud_storage_map.gcloud_command[1] = 'objects'
            else:
                gcloud_storage_map.gcloud_command[1] = 'buckets'
            self.args = url_strings + self.args[:1]
        elif self.sub_command == 'ch':
            if shim_util.HIDDEN_SHIM_MODE(config.get('GSUtil', 'hidden_shim_mode', 'none')) is shim_util.HIDDEN_SHIM_MODE.DRY_RUN:
                self.logger.warning('The shim maps iam ch commands to several gcloud storage commands, which cannot be determined without running gcloud storage.')
            return []
        return super().get_gcloud_storage_args(gcloud_storage_map)

    def _RaiseIfInvalidUrl(self, url):
        if url.IsFileUrl():
            error_msg = 'Invalid Cloud URL "%s".' % url.object_name
            if set(url.object_name).issubset(set('-Rrf')):
                error_msg += ' This resource handle looks like a flag, which must appear before all bindings. See "gsutil help iam ch" for more details.'
            raise CommandException(error_msg)

    def _GetSettingsAndDiffs(self):
        self.continue_on_error = False
        self.recursion_requested = False
        patch_bindings_tuples = []
        if self.sub_opts:
            for o, a in self.sub_opts:
                if o in ['-r', '-R']:
                    self.recursion_requested = True
                elif o == '-f':
                    self.continue_on_error = True
                elif o == '-d':
                    patch_bindings_tuples.append(BindingStringToTuple(False, a))
        url_pattern_strings = []
        it = iter(self.args)
        for token in it:
            if STORAGE_URI_REGEX.match(token) and IsKnownUrlScheme(GetSchemeFromUrlString(token)):
                url_pattern_strings.append(token)
                break
            if token == '-d':
                try:
                    patch_bindings_tuples.append(BindingStringToTuple(False, next(it)))
                except StopIteration:
                    raise CommandException('A -d flag is missing an argument specifying bindings to remove.')
            else:
                patch_bindings_tuples.append(BindingStringToTuple(True, token))
        if not patch_bindings_tuples:
            raise CommandException('Must specify at least one binding.')
        for token in it:
            url_pattern_strings.append(token)
        url_objects = list(map(StorageUrlFromString, url_pattern_strings))
        _RaiseErrorIfUrlsAreMixOfBucketsAndObjects(url_objects, self.recursion_requested)
        return (patch_bindings_tuples, url_objects)

    def _run_ch_subprocess(self, args, stdin=None):
        _, command = self._get_full_gcloud_storage_execution_information(args)
        process = subprocess.run(command, env=self._get_shim_command_environment_variables(), input=stdin, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        if process.returncode != 0:
            if self.continue_on_error:
                self.logger.error(process.stderr)
            else:
                raise CommandException(process.stderr)
        return process

    def run_gcloud_storage(self):
        if self.sub_command != 'ch':
            return super().run_gcloud_storage()
        self.ParseSubOpts()
        bindings_tuples, patterns = self._GetSettingsAndDiffs()
        resource_type = self._get_resource_type(patterns)
        list_settings = []
        if self.recursion_requested:
            list_settings.append('-r')
        return_code = 0
        for url_pattern in patterns:
            self._RaiseIfInvalidUrl(url_pattern)
            if resource_type == 'objects':
                ls_process = self._run_ch_subprocess(['storage', 'ls', '--json'] + list_settings + [str(url_pattern)])
                if ls_process.returncode:
                    return_code = 1
                    continue
                ls_output = json.loads(ls_process.stdout)
                urls = [resource['url'] for resource in ls_output if resource['type'] == 'cloud_object']
            else:
                urls = [str(url_pattern)]
            for url in urls:
                get_process = self._run_ch_subprocess(['storage', resource_type, 'get-iam-policy', url, '--format=json'])
                if get_process.returncode:
                    return_code = 1
                    continue
                policy = json.loads(get_process.stdout)
                bindings = iam_helper.BindingsDictToUpdateDict(policy['bindings'])
                for is_grant, diff in bindings_tuples:
                    diff_dict = iam_helper.BindingsDictToUpdateDict(diff)
                    bindings = PatchBindings(bindings, diff_dict, is_grant)
                policy['bindings'] = [{'role': r, 'members': sorted(list(m))} for r, m in sorted(six.iteritems(bindings))]
                set_process = self._run_ch_subprocess(['storage', resource_type, 'set-iam-policy', url, '-'], stdin=json.dumps(policy, sort_keys=True))
                if set_process.returncode:
                    return_code = 1
        return return_code

    def GetIamHelper(self, storage_url, thread_state=None):
        """Gets an IAM policy for a single, resolved bucket / object URL.

    Args:
      storage_url: A CloudUrl instance with no wildcards, pointing to a
                   specific bucket or object.
      thread_state: CloudApiDelegator instance which is passed from
                    command.WorkerThread.__init__() if the global -m flag is
                    specified. Will use self.gsutil_api if thread_state is set
                    to None.

    Returns:
      Policy instance.
    """
        gsutil_api = GetCloudApiInstance(self, thread_state=thread_state)
        if storage_url.IsBucket():
            policy = gsutil_api.GetBucketIamPolicy(storage_url.bucket_name, provider=storage_url.scheme, fields=['bindings', 'etag'])
        else:
            policy = gsutil_api.GetObjectIamPolicy(storage_url.bucket_name, storage_url.object_name, generation=storage_url.generation, provider=storage_url.scheme, fields=['bindings', 'etag'])
        return policy

    def _GetIam(self, thread_state=None):
        """Gets IAM policy for single bucket or object."""
        pattern = self.args[0]
        matches = PluralityCheckableIterator(self.WildcardIterator(pattern).IterAll(bucket_listing_fields=['name']))
        if matches.IsEmpty():
            raise CommandException('%s matched no URLs' % pattern)
        if matches.HasPlurality():
            raise CommandException('%s matched more than one URL, which is not allowed by the %s command' % (pattern, self.command_name))
        storage_url = StorageUrlFromString(list(matches)[0].url_string)
        policy = self.GetIamHelper(storage_url, thread_state=thread_state)
        policy_json = json.loads(protojson.encode_message(policy))
        policy_str = json.dumps(policy_json, sort_keys=True, separators=(',', ': '), indent=2)
        print(policy_str)

    def _SetIamHelperInternal(self, storage_url, policy, thread_state=None):
        """Sets IAM policy for a single, resolved bucket / object URL.

    Args:
      storage_url: A CloudUrl instance with no wildcards, pointing to a
                   specific bucket or object.
      policy: A Policy object to set on the bucket / object.
      thread_state: CloudApiDelegator instance which is passed from
                    command.WorkerThread.__init__() if the -m flag is
                    specified. Will use self.gsutil_api if thread_state is set
                    to None.

    Raises:
      ServiceException passed from the API call if an HTTP error was returned.
    """
        gsutil_api = GetCloudApiInstance(self, thread_state=thread_state)
        if storage_url.IsBucket():
            gsutil_api.SetBucketIamPolicy(storage_url.bucket_name, policy, provider=storage_url.scheme)
        else:
            gsutil_api.SetObjectIamPolicy(storage_url.bucket_name, storage_url.object_name, policy, generation=storage_url.generation, provider=storage_url.scheme)

    def SetIamHelper(self, storage_url, policy, thread_state=None):
        """Handles the potential exception raised by the internal set function."""
        try:
            self._SetIamHelperInternal(storage_url, policy, thread_state=thread_state)
        except ServiceException:
            if self.continue_on_error:
                self.everything_set_okay = False
            else:
                raise

    def PatchIamHelper(self, storage_url, bindings_tuples, thread_state=None):
        """Patches an IAM policy for a single, resolved bucket / object URL.

    The patch is applied by altering the policy from an IAM get request, and
    setting the new IAM with the specified etag. Because concurrent IAM set
    requests may alter the etag, we may need to retry this operation several
    times before success.

    Args:
      storage_url: A CloudUrl instance with no wildcards, pointing to a
                   specific bucket or object.
      bindings_tuples: A list of BindingsTuple instances.
      thread_state: CloudApiDelegator instance which is passed from
                    command.WorkerThread.__init__() if the -m flag is
                    specified. Will use self.gsutil_api if thread_state is set
                    to None.
    """
        try:
            self._PatchIamHelperInternal(storage_url, bindings_tuples, thread_state=thread_state)
        except ServiceException:
            if self.continue_on_error:
                self.everything_set_okay = False
            else:
                raise
        except IamChOnResourceWithConditionsException as e:
            if self.continue_on_error:
                self.everything_set_okay = False
                self.tried_ch_on_resource_with_conditions = True
                self.logger.debug(e.message)
            else:
                raise CommandException(e.message)

    @Retry(PreconditionException, tries=3, timeout_secs=1.0)
    def _PatchIamHelperInternal(self, storage_url, bindings_tuples, thread_state=None):
        policy = self.GetIamHelper(storage_url, thread_state=thread_state)
        etag, bindings = (policy.etag, policy.bindings)
        for binding in bindings:
            if binding.condition:
                message = 'Could not patch IAM policy for %s.' % storage_url
                message += '\n'
                message += '\n'.join(textwrap.wrap('The resource had conditions present in its IAM policy bindings, which is not supported by "iam ch". %s' % IAM_CH_CONDITIONS_WORKAROUND_MSG))
                raise IamChOnResourceWithConditionsException(message)
        orig_bindings = list(bindings)
        for is_grant, diff in bindings_tuples:
            bindings_dict = iam_helper.BindingsMessageToUpdateDict(bindings)
            diff_dict = iam_helper.BindingsMessageToUpdateDict(diff)
            new_bindings_dict = PatchBindings(bindings_dict, diff_dict, is_grant)
            bindings = [apitools_messages.Policy.BindingsValueListEntry(role=r, members=list(m)) for r, m in six.iteritems(new_bindings_dict)]
        if IsEqualBindings(bindings, orig_bindings):
            self.logger.info('No changes made to %s', storage_url)
            return
        policy = apitools_messages.Policy(bindings=bindings, etag=etag)
        self._SetIamHelperInternal(storage_url, policy, thread_state=thread_state)

    def _PatchIam(self):
        raw_bindings_tuples, url_patterns = self._GetSettingsAndDiffs()
        patch_bindings_tuples = []
        for is_grant, bindings in raw_bindings_tuples:
            bindings_messages = []
            for binding in bindings:
                bindings_message = apitools_messages.Policy.BindingsValueListEntry(members=binding['members'], role=binding['role'])
                bindings_messages.append(bindings_message)
            patch_bindings_tuples.append(BindingsTuple(is_grant=is_grant, bindings=bindings_messages))
        self.everything_set_okay = True
        self.tried_ch_on_resource_with_conditions = False
        threaded_wildcards = []
        for surl in url_patterns:
            try:
                if surl.IsBucket():
                    if self.recursion_requested:
                        surl.object = '*'
                        threaded_wildcards.append(surl.url_string)
                    else:
                        self.PatchIamHelper(surl, patch_bindings_tuples)
                else:
                    threaded_wildcards.append(surl.url_string)
            except AttributeError:
                self._RaiseIfInvalidUrl(surl)
        if threaded_wildcards:
            name_expansion_iterator = NameExpansionIterator(self.command_name, self.debug, self.logger, self.gsutil_api, threaded_wildcards, self.recursion_requested, all_versions=self.all_versions, continue_on_error=self.continue_on_error or self.parallel_operations, bucket_listing_fields=['name'])
            seek_ahead_iterator = SeekAheadNameExpansionIterator(self.command_name, self.debug, self.GetSeekAheadGsutilApi(), threaded_wildcards, self.recursion_requested, all_versions=self.all_versions)
            serialized_bindings_tuples_it = itertools.repeat([SerializeBindingsTuple(t) for t in patch_bindings_tuples])
            self.Apply(_PatchIamWrapper, zip(serialized_bindings_tuples_it, name_expansion_iterator), _PatchIamExceptionHandler, fail_on_error=not self.continue_on_error, seek_ahead_iterator=seek_ahead_iterator)
            self.everything_set_okay &= not GetFailureCount() > 0
        if not self.everything_set_okay:
            msg = 'Some IAM policies could not be patched.'
            if self.tried_ch_on_resource_with_conditions:
                msg += '\n'
                msg += '\n'.join(textwrap.wrap('Some resources had conditions present in their IAM policy bindings, which is not supported by "iam ch". %s' % IAM_CH_CONDITIONS_WORKAROUND_MSG))
            raise CommandException(msg)

    def _SetIam(self):
        """Set IAM policy for given wildcards on the command line."""
        self.continue_on_error = False
        self.recursion_requested = False
        self.all_versions = False
        force_etag = False
        etag = ''
        if self.sub_opts:
            for o, arg in self.sub_opts:
                if o in ['-r', '-R']:
                    self.recursion_requested = True
                elif o == '-f':
                    self.continue_on_error = True
                elif o == '-a':
                    self.all_versions = True
                elif o == '-e':
                    etag = str(arg)
                    force_etag = True
                else:
                    self.RaiseInvalidArgumentException()
        file_url = self.args[0]
        patterns = self.args[1:]
        try:
            with open(file_url, 'r') as fp:
                policy = json.loads(fp.read())
        except IOError:
            raise ArgumentException('Specified IAM policy file "%s" does not exist.' % file_url)
        except ValueError as e:
            self.logger.debug('Invalid IAM policy file, ValueError:\n%s', e)
            raise ArgumentException('Invalid IAM policy file "%s".' % file_url)
        bindings = policy.get('bindings', [])
        if not force_etag:
            etag = policy.get('etag', '')
        policy_json = json.dumps({'bindings': bindings, 'etag': etag, 'version': IAM_POLICY_VERSION})
        try:
            policy = protojson.decode_message(apitools_messages.Policy, policy_json)
        except DecodeError:
            raise ArgumentException('Invalid IAM policy file "%s" or etag "%s".' % (file_url, etag))
        self.everything_set_okay = True
        threaded_wildcards = []
        surls = list(map(StorageUrlFromString, patterns))
        _RaiseErrorIfUrlsAreMixOfBucketsAndObjects(surls, self.recursion_requested)
        for surl in surls:
            print(surl.url_string)
            if surl.IsBucket():
                if self.recursion_requested:
                    surl.object_name = '*'
                    threaded_wildcards.append(surl.url_string)
                else:
                    self.SetIamHelper(surl, policy)
            else:
                threaded_wildcards.append(surl.url_string)
        if threaded_wildcards:
            name_expansion_iterator = NameExpansionIterator(self.command_name, self.debug, self.logger, self.gsutil_api, threaded_wildcards, self.recursion_requested, all_versions=self.all_versions, continue_on_error=self.continue_on_error or self.parallel_operations, bucket_listing_fields=['name'])
            seek_ahead_iterator = SeekAheadNameExpansionIterator(self.command_name, self.debug, self.GetSeekAheadGsutilApi(), threaded_wildcards, self.recursion_requested, all_versions=self.all_versions)
            policy_it = itertools.repeat(protojson.encode_message(policy))
            self.Apply(_SetIamWrapper, zip(policy_it, name_expansion_iterator), _SetIamExceptionHandler, fail_on_error=not self.continue_on_error, seek_ahead_iterator=seek_ahead_iterator)
            self.everything_set_okay &= not GetFailureCount() > 0
        if not self.everything_set_okay:
            raise CommandException('Some IAM policies could not be set.')

    def RunCommand(self):
        """Command entry point for the acl command."""
        action_subcommand = self.args.pop(0)
        self.ParseSubOpts(check_args=True)
        LogCommandParams(sub_opts=self.sub_opts)
        self.def_acl = False
        if action_subcommand == 'get':
            LogCommandParams(subcommands=[action_subcommand])
            self._GetIam()
        elif action_subcommand == 'set':
            LogCommandParams(subcommands=[action_subcommand])
            self._SetIam()
        elif action_subcommand == 'ch':
            LogCommandParams(subcommands=[action_subcommand])
            self._PatchIam()
        else:
            raise CommandException('Invalid subcommand "%s" for the %s command.\nSee "gsutil help iam".' % (action_subcommand, self.command_name))
        return 0