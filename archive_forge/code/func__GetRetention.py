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
def _GetRetention(self):
    """Get Retention Policy for a single bucket."""
    bucket_url, bucket_metadata = self.GetSingleBucketUrlFromArg(self.args[0], bucket_fields=['retentionPolicy'])
    print(RetentionPolicyToString(bucket_metadata.retentionPolicy, bucket_url))
    return 0