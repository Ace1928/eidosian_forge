from __future__ import absolute_import
from __future__ import print_function
import getopt
import textwrap
from gslib import metrics
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.commands import ubla
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.help_provider import CreateHelpText
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import NO_MAX
from gslib.utils.text_util import InsistOnOrOff
def _GetBucketPolicyOnly(self, blr):
    """Gets the Bucket Policy Only setting for a bucket."""
    self._ValidateBucketListingRefAndReturnBucketName(blr)
    bucket_url = blr.storage_url
    bucket_metadata = self.gsutil_api.GetBucket(bucket_url.bucket_name, fields=['iamConfiguration'], provider=bucket_url.scheme)
    iam_config = bucket_metadata.iamConfiguration
    bucket_policy_only = iam_config.bucketPolicyOnly
    fields = {'bucket': str(bucket_url).rstrip('/'), 'enabled': bucket_policy_only.enabled}
    locked_time_line = ''
    if bucket_policy_only.lockedTime:
        fields['locked_time'] = bucket_policy_only.lockedTime
        locked_time_line = '  LockedTime: {locked_time}\n'
    if bucket_policy_only:
        print(('Bucket Policy Only setting for {bucket}:\n  Enabled: {enabled}\n' + locked_time_line).format(**fields))