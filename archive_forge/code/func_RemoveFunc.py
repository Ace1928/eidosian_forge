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
def RemoveFunc(self, name_expansion_result, thread_state=None):
    gsutil_api = GetCloudApiInstance(self, thread_state=thread_state)
    exp_src_url = name_expansion_result.expanded_storage_url
    self.logger.info('Removing %s...', exp_src_url)
    try:
        gsutil_api.DeleteObject(exp_src_url.bucket_name, exp_src_url.object_name, preconditions=self.preconditions, generation=exp_src_url.generation, provider=exp_src_url.scheme)
    except NotFoundException as e:
        self.logger.info('Cannot find %s', exp_src_url)
        DecrementFailureCount()
    _PutToQueueWithTimeout(gsutil_api.status_queue, MetadataMessage(message_time=time.time()))