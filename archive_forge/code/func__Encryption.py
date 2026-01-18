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