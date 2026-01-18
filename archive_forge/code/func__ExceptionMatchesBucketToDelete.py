from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import time
from gslib.cloud_api import BucketNotFoundException
from gslib.cloud_api import NotEmptyException
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import ServiceException
from gslib.command import Command
from gslib.command import DecrementFailureCount
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_PREFIX
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.name_expansion import NameExpansionIterator
from gslib.name_expansion import SeekAheadNameExpansionIterator
from gslib.storage_url import StorageUrlFromString
from gslib.thread_message import MetadataMessage
from gslib.utils import constants
from gslib.utils import parallelism_framework_util
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.retry_util import Retry
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.system_util import StdinIterator
from gslib.utils.translation_helper import PreconditionsFromHeaders
def _ExceptionMatchesBucketToDelete(bucket_strings_to_delete, e):
    """Returns True if the exception matches a bucket slated for deletion.

  A recursive delete call on an empty bucket will raise an exception when
  listing its objects, but if we plan to delete the bucket that shouldn't
  result in a user-visible error.

  Args:
    bucket_strings_to_delete: Buckets slated for recursive deletion.
    e: Exception to check.

  Returns:
    True if the exception was a no-URLs-matched exception and it matched
    one of bucket_strings_to_delete, None otherwise.
  """
    if bucket_strings_to_delete:
        msg = NO_URLS_MATCHED_TARGET % ''
        if msg in str(e):
            parts = str(e).split(msg)
            return len(parts) == 2 and parts[1] in bucket_strings_to_delete