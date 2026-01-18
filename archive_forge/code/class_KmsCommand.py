from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import getopt
import textwrap
from gslib import metrics
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import ServiceException
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.help_provider import CreateHelpText
from gslib.kms_api import KmsApi
from gslib.project_id import PopulateProjectId
from gslib.third_party.kms_apitools.cloudkms_v1_messages import Binding
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import text_util
from gslib.utils.constants import NO_MAX
from gslib.utils.encryption_helper import ValidateCMEK
from gslib.utils.retry_util import Retry
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
class KmsCommand(Command):
    """Implements of gsutil kms command."""
    command_spec = Command.CreateCommandSpec('kms', usage_synopsis=_SYNOPSIS, min_args=1, max_args=NO_MAX, supported_sub_args='dk:p:w', file_url_ok=False, provider_url_ok=False, urls_start_arg=1, gs_api_support=[ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments={'authorize': [], 'encryption': [CommandArgument.MakeNCloudBucketURLsArgument(1)], 'serviceaccount': []})
    help_spec = Command.HelpSpec(help_name='kms', help_name_aliases=[], help_type='command_help', help_one_line_summary='Configure Cloud KMS encryption', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={'authorize': _authorize_help_text, 'encryption': _encryption_help_text, 'serviceaccount': _serviceaccount_help_text})
    gcloud_storage_map = GcloudStorageMap(gcloud_command={'authorize': _AUTHORIZE_COMMAND, 'serviceaccount': _SERVICEACCOUNT_COMMAND}, flag_map={})

    def get_gcloud_storage_args(self):
        if self.args[0] == 'encryption':
            gcloud_storage_map = GcloudStorageMap(gcloud_command={'encryption': GcloudStorageMap(gcloud_command=['storage', 'buckets'], flag_map={'-d': GcloudStorageFlag('--clear-default-encryption-key'), '-k': GcloudStorageFlag('--default-encryption-key'), '-w': GcloudStorageFlag('')})}, flag_map={})
            if '-d' in self.args or '-k' in self.args:
                gcloud_storage_map.gcloud_command['encryption'].gcloud_command += ['update']
            else:
                gcloud_storage_map.gcloud_command['encryption'].gcloud_command += ['describe', '--format=value[separator=": "](name, encryption.defaultKmsKeyName.yesno(no="No default encryption key."))', '--raw']
        else:
            gcloud_storage_map = KmsCommand.gcloud_storage_map
        return super().get_gcloud_storage_args(gcloud_storage_map)

    def _GatherSubOptions(self, subcommand_name):
        self.CheckArguments()
        self.clear_kms_key = False
        self.kms_key = None
        self.warn_on_key_authorize_failure = False
        if self.sub_opts:
            for o, a in self.sub_opts:
                if o == '-p':
                    self.project_id = a
                elif o == '-k':
                    self.kms_key = a
                    ValidateCMEK(self.kms_key)
                elif o == '-d':
                    self.clear_kms_key = True
                elif o == '-w':
                    self.warn_on_key_authorize_failure = True
        if self.warn_on_key_authorize_failure and (self.subcommand_name != 'encryption' or not self.kms_key):
            raise CommandException('\n'.join(textwrap.wrap('The "-w" option should only be specified for the "encryption" subcommand and must be used with the "-k" option.')))
        if not self.project_id:
            self.project_id = PopulateProjectId(None)

    @Retry(ServiceException, tries=3, timeout_secs=1)
    def _AuthorizeProject(self, project_id, kms_key):
        """Authorizes a project's service account to be used with a KMS key.

    Authorizes the Cloud Storage-owned service account for project_id to be used
    with kms_key.

    Args:
      project_id: (str) Project id string (not number).
      kms_key: (str) Fully qualified resource name for the KMS key.

    Returns:
      (str, bool) A 2-tuple consisting of:
      1) The email address for the service account associated with the project,
         which is authorized to encrypt/decrypt with the specified key.
      2) A bool value - True if we had to grant the service account permission
         to encrypt/decrypt with the given key; False if the required permission
         was already present.
    """
        service_account = self.gsutil_api.GetProjectServiceAccount(project_id, provider='gs').email_address
        kms_api = KmsApi(logger=self.logger)
        self.logger.debug('Getting IAM policy for %s', kms_key)
        try:
            policy = kms_api.GetKeyIamPolicy(kms_key)
            self.logger.debug('Current policy is %s', policy)
            added_new_binding = False
            binding = Binding(role='roles/cloudkms.cryptoKeyEncrypterDecrypter', members=['serviceAccount:%s' % service_account])
            if binding not in policy.bindings:
                policy.bindings.append(binding)
                kms_api.SetKeyIamPolicy(kms_key, policy)
                added_new_binding = True
            return (service_account, added_new_binding)
        except AccessDeniedException:
            if self.warn_on_key_authorize_failure:
                text_util.print_to_fd('\n'.join(textwrap.wrap('Warning: Check that your Cloud Platform project\'s service account has the "cloudkms.cryptoKeyEncrypterDecrypter" role for the specified key. Without this role, you may not be able to encrypt or decrypt objects using the key which will prevent you from uploading or downloading objects.')))
                return (service_account, False)
            else:
                raise

    def _Authorize(self):
        self._GatherSubOptions('authorize')
        if not self.kms_key:
            raise CommandException('%s %s requires a key to be specified with -k' % (self.command_name, self.subcommand_name))
        _, newly_authorized = self._AuthorizeProject(self.project_id, self.kms_key)
        if newly_authorized:
            print('Authorized project %s to encrypt and decrypt with key:\n%s' % (self.project_id, self.kms_key))
        else:
            print('Project %s was already authorized to encrypt and decrypt with key:\n%s.' % (self.project_id, self.kms_key))
        return 0

    def _EncryptionClearKey(self, bucket_metadata, bucket_url):
        """Clears the defaultKmsKeyName on a Cloud Storage bucket.

    Args:
      bucket_metadata: (apitools_messages.Bucket) Metadata for the given bucket.
      bucket_url: (gslib.storage_url.StorageUrl) StorageUrl of the given bucket.
    """
        bucket_metadata.encryption = apitools_messages.Bucket.EncryptionValue()
        print('Clearing default encryption key for %s...' % str(bucket_url).rstrip('/'))
        self.gsutil_api.PatchBucket(bucket_url.bucket_name, bucket_metadata, fields=['encryption'], provider=bucket_url.scheme)

    def _EncryptionSetKey(self, bucket_metadata, bucket_url, svc_acct_for_project_num):
        """Sets defaultKmsKeyName on a Cloud Storage bucket.

    Args:
      bucket_metadata: (apitools_messages.Bucket) Metadata for the given bucket.
      bucket_url: (gslib.storage_url.StorageUrl) StorageUrl of the given bucket.
      svc_acct_for_project_num: (Dict[int, str]) Mapping of project numbers to
          their corresponding service account.
    """
        bucket_project_number = bucket_metadata.projectNumber
        try:
            service_account, newly_authorized = (svc_acct_for_project_num[bucket_project_number], False)
        except KeyError:
            service_account, newly_authorized = self._AuthorizeProject(bucket_project_number, self.kms_key)
            svc_acct_for_project_num[bucket_project_number] = service_account
        if newly_authorized:
            text_util.print_to_fd('Authorized service account %s to use key:\n%s' % (service_account, self.kms_key))
        bucket_metadata.encryption = apitools_messages.Bucket.EncryptionValue(defaultKmsKeyName=self.kms_key)
        print('Setting default KMS key for bucket %s...' % str(bucket_url).rstrip('/'))
        self.gsutil_api.PatchBucket(bucket_url.bucket_name, bucket_metadata, fields=['encryption'], provider=bucket_url.scheme)

    def _Encryption(self):
        self._GatherSubOptions('encryption')
        svc_acct_for_project_num = {}

        def _EncryptionForBucket(blr):
            """Set, clear, or get the defaultKmsKeyName for a bucket."""
            bucket_url = blr.storage_url
            if bucket_url.scheme != 'gs':
                raise CommandException('The %s command can only be used with gs:// bucket URLs.' % self.command_name)
            bucket_metadata = self.gsutil_api.GetBucket(bucket_url.bucket_name, fields=['encryption', 'projectNumber'], provider=bucket_url.scheme)
            if self.clear_kms_key:
                self._EncryptionClearKey(bucket_metadata, bucket_url)
                return 0
            if self.kms_key:
                self._EncryptionSetKey(bucket_metadata, bucket_url, svc_acct_for_project_num)
                return 0
            bucket_url_string = str(bucket_url).rstrip('/')
            if bucket_metadata.encryption and bucket_metadata.encryption.defaultKmsKeyName:
                print('Default encryption key for %s:\n%s' % (bucket_url_string, bucket_metadata.encryption.defaultKmsKeyName))
            else:
                print('Bucket %s has no default encryption key' % bucket_url_string)
            return 0
        some_matched = False
        url_args = self.args
        if not url_args:
            self.RaiseWrongNumberOfArgumentsException()
        for url_str in url_args:
            bucket_iter = self.GetBucketUrlIterFromArg(url_str)
            for bucket_listing_ref in bucket_iter:
                some_matched = True
                _EncryptionForBucket(bucket_listing_ref)
        if not some_matched:
            raise CommandException(NO_URLS_MATCHED_TARGET % list(url_args))
        return 0

    def _ServiceAccount(self):
        self.CheckArguments()
        if not self.args:
            self.args = ['gs://']
        if self.sub_opts:
            for o, a in self.sub_opts:
                if o == '-p':
                    self.project_id = a
        if not self.project_id:
            self.project_id = PopulateProjectId(None)
        self.logger.debug('Checking service account for project %s', self.project_id)
        service_account = self.gsutil_api.GetProjectServiceAccount(self.project_id, provider='gs').email_address
        print(service_account)
        return 0

    def _RunSubCommand(self, func):
        try:
            self.sub_opts, self.args = getopt.getopt(self.args, self.command_spec.supported_sub_args)
            metrics.LogCommandParams(sub_opts=self.sub_opts)
            return func(self)
        except getopt.GetoptError:
            self.RaiseInvalidArgumentException()

    def RunCommand(self):
        """Command entry point for the kms command."""
        if self.gsutil_api.GetApiSelector(provider='gs') != ApiSelector.JSON:
            raise CommandException('\n'.join(textwrap.wrap('The "%s" command can only be used with the GCS JSON API. If you have only supplied hmac credentials in your boto file, please instead supply a credential type that can be used with the JSON API.' % self.command_name)))

    def RunCommand(self):
        """Command entry point for the kms command."""
        if self.gsutil_api.GetApiSelector(provider='gs') != ApiSelector.JSON:
            raise CommandException('\n'.join(textwrap.wrap('The "%s" command can only be used with the GCS JSON API, which cannot use HMAC credentials. Please supply a credential type that is compatible with the JSON API (e.g. OAuth2) in your boto config file.' % self.command_name)))
        method_for_subcommand = {'authorize': KmsCommand._Authorize, 'encryption': KmsCommand._Encryption, 'serviceaccount': KmsCommand._ServiceAccount}
        self.subcommand_name = self.args.pop(0)
        if self.subcommand_name in method_for_subcommand:
            metrics.LogCommandParams(subcommands=[self.subcommand_name])
            return self._RunSubCommand(method_for_subcommand[self.subcommand_name])
        else:
            raise CommandException('Invalid subcommand "%s" for the %s command.' % (self.subcommand_name, self.command_name))