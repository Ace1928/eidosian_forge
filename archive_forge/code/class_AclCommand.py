from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import encoding
from gslib import metrics
from gslib import gcs_json_api
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import BadRequestException
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import Preconditions
from gslib.cloud_api import ServiceException
from gslib.command import Command
from gslib.command import SetAclExceptionHandler
from gslib.command import SetAclFuncWrapper
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.help_provider import CreateHelpText
from gslib.storage_url import StorageUrlFromString
from gslib.storage_url import UrlsAreForSingleProvider
from gslib.storage_url import RaiseErrorIfUrlsAreMixOfBucketsAndObjects
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import acl_helper
from gslib.utils.constants import NO_MAX
from gslib.utils.retry_util import Retry
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
class AclCommand(Command):
    """Implementation of gsutil acl command."""
    command_spec = Command.CreateCommandSpec('acl', command_name_aliases=['getacl', 'setacl', 'chacl'], usage_synopsis=_SYNOPSIS, min_args=2, max_args=NO_MAX, supported_sub_args='afRrg:u:d:p:', file_url_ok=False, provider_url_ok=False, urls_start_arg=1, gs_api_support=[ApiSelector.XML, ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments={'set': [CommandArgument.MakeFileURLOrCannedACLArgument(), CommandArgument.MakeZeroOrMoreCloudURLsArgument()], 'get': [CommandArgument.MakeNCloudURLsArgument(1)], 'ch': [CommandArgument.MakeZeroOrMoreCloudURLsArgument()]})
    help_spec = Command.HelpSpec(help_name='acl', help_name_aliases=['getacl', 'setacl', 'chmod', 'chacl'], help_type='command_help', help_one_line_summary='Get, set, or change bucket and/or object ACLs', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={'get': _get_help_text, 'set': _set_help_text, 'ch': _ch_help_text})

    def _get_shim_command_group(self):
        object_or_bucket_urls = [StorageUrlFromString(url) for url in self.args]
        recurse = False
        for flag_key, _ in self.sub_opts:
            if flag_key in ('-r', '-R'):
                recurse = True
                break
        RaiseErrorIfUrlsAreMixOfBucketsAndObjects(object_or_bucket_urls, recurse)
        if object_or_bucket_urls[0].IsBucket() and (not recurse):
            return 'buckets'
        else:
            return 'objects'

    def get_gcloud_storage_args(self):
        sub_command = self.args.pop(0)
        if sub_command == 'get':
            if StorageUrlFromString(self.args[0]).IsObject():
                command_group = 'objects'
            else:
                command_group = 'buckets'
            gcloud_storage_map = GcloudStorageMap(gcloud_command=['storage', command_group, 'describe', '--format=multi(acl:format=json)', '--raw'], flag_map={})
        elif sub_command == 'set':
            self.ParseSubOpts()
            acl_file_or_predefined_acl = self.args.pop(0)
            if os.path.isfile(acl_file_or_predefined_acl):
                acl_flag = '--acl-file=' + acl_file_or_predefined_acl
            else:
                if acl_file_or_predefined_acl in gcs_json_api.FULL_PREDEFINED_ACL_XML_TO_JSON_TRANSLATION:
                    predefined_acl = gcs_json_api.FULL_PREDEFINED_ACL_XML_TO_JSON_TRANSLATION[acl_file_or_predefined_acl]
                else:
                    predefined_acl = acl_file_or_predefined_acl
                acl_flag = '--predefined-acl=' + predefined_acl
            command_group = self._get_shim_command_group()
            gcloud_storage_map = GcloudStorageMap(gcloud_command=['storage', command_group, 'update'] + [acl_flag], flag_map={'-a': GcloudStorageFlag('--all-versions'), '-f': GcloudStorageFlag('--continue-on-error'), '-R': GcloudStorageFlag('--recursive'), '-r': GcloudStorageFlag('--recursive')})
        elif sub_command == 'ch':
            self.ParseSubOpts()
            self.sub_opts = acl_helper.translate_sub_opts_for_shim(self.sub_opts)
            command_group = self._get_shim_command_group()
            gcloud_storage_map = GcloudStorageMap(gcloud_command=['storage', command_group, 'update'], flag_map={'-g': GcloudStorageFlag('--add-acl-grant'), '-p': GcloudStorageFlag('--add-acl-grant'), '-u': GcloudStorageFlag('--add-acl-grant'), '-d': GcloudStorageFlag('--remove-acl-grant'), '-a': GcloudStorageFlag('--all-versions'), '-f': GcloudStorageFlag('--continue-on-error'), '-R': GcloudStorageFlag('--recursive'), '-r': GcloudStorageFlag('--recursive')})
        return super().get_gcloud_storage_args(gcloud_storage_map)

    def _CalculateUrlsStartArg(self):
        if not self.args:
            self.RaiseWrongNumberOfArgumentsException()
        if self.args[0].lower() == 'set' or self.command_alias_used == 'setacl':
            return 1
        else:
            return 0

    def _SetAcl(self):
        """Parses options and sets ACLs on the specified buckets/objects."""
        self.continue_on_error = False
        if self.sub_opts:
            for o, unused_a in self.sub_opts:
                if o == '-a':
                    self.all_versions = True
                elif o == '-f':
                    self.continue_on_error = True
                elif o == '-r' or o == '-R':
                    self.recursion_requested = True
                else:
                    self.RaiseInvalidArgumentException()
        try:
            self.SetAclCommandHelper(SetAclFuncWrapper, SetAclExceptionHandler)
        except AccessDeniedException as unused_e:
            self._WarnServiceAccounts()
            raise
        if not self.everything_set_okay:
            raise CommandException('ACLs for some objects could not be set.')

    def _ChAcl(self):
        """Parses options and changes ACLs on the specified buckets/objects."""
        self.parse_versions = True
        self.changes = []
        self.continue_on_error = False
        if self.sub_opts:
            for o, a in self.sub_opts:
                if o == '-f':
                    self.continue_on_error = True
                elif o == '-g':
                    if 'gserviceaccount.com' in a:
                        raise CommandException('Service accounts are considered users, not groups; please use "gsutil acl ch -u" instead of "gsutil acl ch -g"')
                    self.changes.append(acl_helper.AclChange(a, scope_type=acl_helper.ChangeType.GROUP))
                elif o == '-p':
                    self.changes.append(acl_helper.AclChange(a, scope_type=acl_helper.ChangeType.PROJECT))
                elif o == '-u':
                    self.changes.append(acl_helper.AclChange(a, scope_type=acl_helper.ChangeType.USER))
                elif o == '-d':
                    self.changes.append(acl_helper.AclDel(a))
                elif o == '-r' or o == '-R':
                    self.recursion_requested = True
                else:
                    self.RaiseInvalidArgumentException()
        if not self.changes:
            raise CommandException('Please specify at least one access change with the -g, -u, or -d flags')
        if not UrlsAreForSingleProvider(self.args) or StorageUrlFromString(self.args[0]).scheme != 'gs':
            raise CommandException('The "{0}" command can only be used with gs:// URLs'.format(self.command_name))
        self.everything_set_okay = True
        self.ApplyAclFunc(_ApplyAclChangesWrapper, _ApplyExceptionHandler, self.args, object_fields=['acl', 'generation', 'metageneration'])
        if not self.everything_set_okay:
            raise CommandException('ACLs for some objects could not be set.')

    def _RaiseForAccessDenied(self, url):
        self._WarnServiceAccounts()
        raise CommandException('Failed to set acl for %s. Please ensure you have OWNER-role access to this resource.' % url)

    @Retry(ServiceException, tries=3, timeout_secs=1)
    def ApplyAclChanges(self, name_expansion_result, thread_state=None):
        """Applies the changes in self.changes to the provided URL.

    Args:
      name_expansion_result: NameExpansionResult describing the target object.
      thread_state: If present, gsutil Cloud API instance to apply the changes.
    """
        if thread_state:
            gsutil_api = thread_state
        else:
            gsutil_api = self.gsutil_api
        url = name_expansion_result.expanded_storage_url
        if url.IsBucket():
            bucket = gsutil_api.GetBucket(url.bucket_name, provider=url.scheme, fields=['acl', 'metageneration'])
            current_acl = bucket.acl
        elif url.IsObject():
            gcs_object = encoding.JsonToMessage(apitools_messages.Object, name_expansion_result.expanded_result)
            current_acl = gcs_object.acl
        if not current_acl:
            self._RaiseForAccessDenied(url)
        if self._ApplyAclChangesAndReturnChangeCount(url, current_acl) == 0:
            self.logger.info('No changes to %s', url)
            return
        try:
            if url.IsBucket():
                preconditions = Preconditions(meta_gen_match=bucket.metageneration)
                bucket_metadata = apitools_messages.Bucket(acl=current_acl)
                gsutil_api.PatchBucket(url.bucket_name, bucket_metadata, preconditions=preconditions, provider=url.scheme, fields=['id'])
            else:
                preconditions = Preconditions(gen_match=gcs_object.generation, meta_gen_match=gcs_object.metageneration)
                object_metadata = apitools_messages.Object(acl=current_acl)
                try:
                    gsutil_api.PatchObjectMetadata(url.bucket_name, url.object_name, object_metadata, preconditions=preconditions, provider=url.scheme, generation=url.generation, fields=['id'])
                except PreconditionException as e:
                    self._RefetchObjectMetadataAndApplyAclChanges(url, gsutil_api)
            self.logger.info('Updated ACL on %s', url)
        except BadRequestException as e:
            raise CommandException('Received bad request from server: %s' % str(e))
        except AccessDeniedException:
            self._RaiseForAccessDenied(url)
        except PreconditionException as e:
            if url.IsObject():
                raise CommandException(str(e))
            raise e

    @Retry(PreconditionException, tries=3, timeout_secs=1)
    def _RefetchObjectMetadataAndApplyAclChanges(self, url, gsutil_api):
        """Reattempts object ACL changes after a PreconditionException."""
        gcs_object = gsutil_api.GetObjectMetadata(url.bucket_name, url.object_name, provider=url.scheme, fields=['acl', 'generation', 'metageneration'])
        current_acl = gcs_object.acl
        if self._ApplyAclChangesAndReturnChangeCount(url, current_acl) == 0:
            self.logger.info('No changes to %s', url)
            return
        object_metadata = apitools_messages.Object(acl=current_acl)
        preconditions = Preconditions(gen_match=gcs_object.generation, meta_gen_match=gcs_object.metageneration)
        gsutil_api.PatchObjectMetadata(url.bucket_name, url.object_name, object_metadata, preconditions=preconditions, provider=url.scheme, generation=gcs_object.generation, fields=['id'])

    def _ApplyAclChangesAndReturnChangeCount(self, storage_url, acl_message):
        modification_count = 0
        for change in self.changes:
            modification_count += change.Execute(storage_url, acl_message, 'acl', self.logger)
        return modification_count

    def RunCommand(self):
        """Command entry point for the acl command."""
        action_subcommand = self.args.pop(0)
        self.ParseSubOpts(check_args=True)
        metrics.LogCommandParams(sub_opts=self.sub_opts)
        self.def_acl = False
        if action_subcommand == 'get':
            metrics.LogCommandParams(subcommands=[action_subcommand])
            self.GetAndPrintAcl(self.args[0])
        elif action_subcommand == 'set':
            metrics.LogCommandParams(subcommands=[action_subcommand])
            self._SetAcl()
        elif action_subcommand in ('ch', 'change'):
            metrics.LogCommandParams(subcommands=[action_subcommand])
            self._ChAcl()
        else:
            raise CommandException('Invalid subcommand "%s" for the %s command.\nSee "gsutil help acl".' % (action_subcommand, self.command_name))
        return 0