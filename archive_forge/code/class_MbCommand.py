from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
import textwrap
from gslib.cloud_api import AccessDeniedException, BadRequestException
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.commands.rpo import VALID_RPO_VALUES
from gslib.commands.rpo import VALID_RPO_VALUES_STRING
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import InvalidUrlError
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import NO_MAX
from gslib.utils.retention_util import RetentionInSeconds
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.text_util import InsistAscii
from gslib.utils.text_util import InsistOnOrOff
from gslib.utils.text_util import NormalizeStorageClass
from gslib.utils.encryption_helper import ValidateCMEK
class MbCommand(Command):
    """Implementation of gsutil mb command."""
    command_spec = Command.CreateCommandSpec('mb', command_name_aliases=['makebucket', 'createbucket', 'md', 'mkdir'], usage_synopsis=_SYNOPSIS, min_args=1, max_args=NO_MAX, supported_sub_args='b:c:l:p:s:k:', supported_private_args=['autoclass', 'retention=', 'pap=', 'placement=', 'rpo='], file_url_ok=False, provider_url_ok=False, urls_start_arg=0, gs_api_support=[ApiSelector.XML, ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments=[CommandArgument.MakeZeroOrMoreCloudBucketURLsArgument()])
    help_spec = Command.HelpSpec(help_name='mb', help_name_aliases=['createbucket', 'makebucket', 'md', 'mkdir', 'location', 'dra', 'dras', 'reduced_availability', 'durable_reduced_availability', 'rr', 'reduced_redundancy', 'standard', 'storage class', 'nearline', 'nl'], help_type='command_help', help_one_line_summary='Make buckets', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={})
    gcloud_storage_map = GcloudStorageMap(gcloud_command=['storage', 'buckets', 'create'], flag_map={'-b': GcloudStorageFlag({'on': '--uniform-bucket-level-access', 'off': None}), '-c': GcloudStorageFlag('--default-storage-class'), '-k': GcloudStorageFlag('--default-encryption-key'), '-l': GcloudStorageFlag('--location'), '-p': GcloudStorageFlag('--project'), '--pap': GcloudStorageFlag({'enforced': '--public-access-prevention', 'inherited': None}), '--placement': GcloudStorageFlag('--placement'), _RETENTION_FLAG: GcloudStorageFlag('--retention-period'), '--rpo': GcloudStorageFlag('--recovery-point-objective')})

    def get_gcloud_storage_args(self):
        retention_arg_idx = 0
        while retention_arg_idx < len(self.sub_opts):
            if self.sub_opts[retention_arg_idx][0] == _RETENTION_FLAG:
                break
            retention_arg_idx += 1
        if retention_arg_idx < len(self.sub_opts):
            self.sub_opts[retention_arg_idx] = (_RETENTION_FLAG, str(RetentionInSeconds(self.sub_opts[retention_arg_idx][1])) + 's')
        return super().get_gcloud_storage_args(MbCommand.gcloud_storage_map)

    def RunCommand(self):
        """Command entry point for the mb command."""
        autoclass = False
        bucket_policy_only = None
        kms_key = None
        location = None
        storage_class = None
        seconds = None
        placements = None
        public_access_prevention = None
        rpo = None
        json_only_flags_in_command = []
        if self.sub_opts:
            for o, a in self.sub_opts:
                if o == '--autoclass':
                    autoclass = True
                    json_only_flags_in_command.append(o)
                elif o == '-k':
                    kms_key = a
                    ValidateCMEK(kms_key)
                    json_only_flags_in_command.append(o)
                elif o == '-l':
                    location = a
                elif o == '-p':
                    InsistAscii(a, 'Invalid non-ASCII character found in project ID')
                    self.project_id = a
                elif o == '-c' or o == '-s':
                    storage_class = NormalizeStorageClass(a)
                elif o == _RETENTION_FLAG:
                    seconds = RetentionInSeconds(a)
                elif o == '--rpo':
                    rpo = a.strip()
                    if rpo not in VALID_RPO_VALUES:
                        raise CommandException('Invalid value for --rpo. Must be one of: {}, provided: {}'.format(VALID_RPO_VALUES_STRING, a))
                    json_only_flags_in_command.append(o)
                elif o == '-b':
                    InsistOnOrOff(a, 'Only on and off values allowed for -b option')
                    bucket_policy_only = a == 'on'
                    json_only_flags_in_command.append(o)
                elif o == '--pap':
                    public_access_prevention = a
                    json_only_flags_in_command.append(o)
                elif o == '--placement':
                    placements = a.split(',')
                    if len(placements) != 2:
                        raise CommandException('Please specify two regions separated by comma without space. Specified: {}'.format(a))
                    json_only_flags_in_command.append(o)
        bucket_metadata = apitools_messages.Bucket(location=location, rpo=rpo, storageClass=storage_class)
        if autoclass:
            bucket_metadata.autoclass = apitools_messages.Bucket.AutoclassValue(enabled=autoclass)
        if bucket_policy_only or public_access_prevention:
            bucket_metadata.iamConfiguration = IamConfigurationValue()
            iam_config = bucket_metadata.iamConfiguration
            if bucket_policy_only:
                iam_config.bucketPolicyOnly = BucketPolicyOnlyValue()
                iam_config.bucketPolicyOnly.enabled = bucket_policy_only
            if public_access_prevention:
                iam_config.publicAccessPrevention = public_access_prevention
        if kms_key:
            encryption = apitools_messages.Bucket.EncryptionValue()
            encryption.defaultKmsKeyName = kms_key
            bucket_metadata.encryption = encryption
        if placements:
            placement_config = apitools_messages.Bucket.CustomPlacementConfigValue()
            placement_config.dataLocations = placements
            bucket_metadata.customPlacementConfig = placement_config
        for bucket_url_str in self.args:
            bucket_url = StorageUrlFromString(bucket_url_str)
            if seconds is not None:
                if bucket_url.scheme != 'gs':
                    raise CommandException('Retention policy can only be specified for GCS buckets.')
                retention_policy = apitools_messages.Bucket.RetentionPolicyValue(retentionPeriod=seconds)
                bucket_metadata.retentionPolicy = retention_policy
            if json_only_flags_in_command and self.gsutil_api.GetApiSelector(bucket_url.scheme) != ApiSelector.JSON:
                raise CommandException('The {} option(s) can only be used for GCS Buckets with the JSON API'.format(', '.join(json_only_flags_in_command)))
            if not bucket_url.IsBucket():
                raise CommandException('The mb command requires a URL that specifies a bucket.\n"%s" is not valid.' % bucket_url)
            if not BUCKET_NAME_RE.match(bucket_url.bucket_name) or TOO_LONG_DNS_NAME_COMP.search(bucket_url.bucket_name):
                raise InvalidUrlError('Invalid bucket name in URL "%s"' % bucket_url.bucket_name)
            self.logger.info('Creating %s...', bucket_url)
            try:
                self.gsutil_api.CreateBucket(bucket_url.bucket_name, project_id=self.project_id, metadata=bucket_metadata, provider=bucket_url.scheme)
            except AccessDeniedException as e:
                message = e.reason
                if 'key' in message:
                    message += ' To authorize, run:\n  gsutil kms authorize'
                    message += ' \\\n    -k %s' % kms_key
                    if self.project_id:
                        message += ' \\\n    -p %s' % self.project_id
                    raise CommandException(message)
                else:
                    raise
            except BadRequestException as e:
                if e.status == 400 and e.reason == 'DotfulBucketNameNotUnderTld' and (bucket_url.scheme == 'gs'):
                    bucket_name = bucket_url.bucket_name
                    final_comp = bucket_name[bucket_name.rfind('.') + 1:]
                    raise CommandException('\n'.join(textwrap.wrap('Buckets with "." in the name must be valid DNS names. The bucket you are attempting to create (%s) is not a valid DNS name, because the final component (%s) is not currently a valid part of the top-level DNS tree.' % (bucket_name, final_comp))))
                else:
                    raise
        return 0