from __future__ import absolute_import
from __future__ import print_function
import textwrap
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
def _GetRpo(self, blr):
    """Gets the rpo setting for a bucket."""
    bucket_url = blr.storage_url
    bucket_metadata = self.gsutil_api.GetBucket(bucket_url.bucket_name, fields=['rpo'], provider=bucket_url.scheme)
    rpo = bucket_metadata.rpo
    bucket = str(bucket_url).rstrip('/')
    print('%s: %s' % (bucket, rpo))