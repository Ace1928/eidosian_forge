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
def ObjectUpdateMetadataFunc(self, patch_obj_metadata, log_template, name_expansion_result, thread_state=None):
    """Updates metadata on an object using PatchObjectMetadata.

    Args:
      patch_obj_metadata: Metadata changes that should be applied to the
                          existing object.
      log_template: The log template that should be printed for each object.
      name_expansion_result: NameExpansionResult describing target object.
      thread_state: gsutil Cloud API instance to use for the operation.
    """
    gsutil_api = GetCloudApiInstance(self, thread_state=thread_state)
    exp_src_url = name_expansion_result.expanded_storage_url
    self.logger.info(log_template, exp_src_url)
    cloud_obj_metadata = encoding.JsonToMessage(apitools_messages.Object, name_expansion_result.expanded_result)
    preconditions = Preconditions(gen_match=self.preconditions.gen_match, meta_gen_match=self.preconditions.meta_gen_match)
    if preconditions.gen_match is None:
        preconditions.gen_match = cloud_obj_metadata.generation
    if preconditions.meta_gen_match is None:
        preconditions.meta_gen_match = cloud_obj_metadata.metageneration
    gsutil_api.PatchObjectMetadata(exp_src_url.bucket_name, exp_src_url.object_name, patch_obj_metadata, generation=exp_src_url.generation, preconditions=preconditions, provider=exp_src_url.scheme, fields=['id'])
    PutToQueueWithTimeout(gsutil_api.status_queue, MetadataMessage(message_time=time.time()))