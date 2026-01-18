from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import codecs
from collections import namedtuple
import copy
import getopt
import json
import logging
import os
import signal
import sys
import textwrap
import threading
import time
import traceback
import boto
from boto.storage_uri import StorageUri
import gslib
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import ServiceException
from gslib.cloud_api_delegator import CloudApiDelegator
from gslib.cs_api_map import ApiSelector
from gslib.cs_api_map import GsutilApiMapFactory
from gslib.exception import CommandException
from gslib.help_provider import HelpProvider
from gslib.metrics import CaptureThreadStatException
from gslib.metrics import LogPerformanceSummaryParams
from gslib.name_expansion import CopyObjectInfo
from gslib.name_expansion import CopyObjectsIterator
from gslib.name_expansion import NameExpansionIterator
from gslib.name_expansion import NameExpansionResult
from gslib.name_expansion import SeekAheadNameExpansionIterator
from gslib.plurality_checkable_iterator import PluralityCheckableIterator
from gslib.seek_ahead_thread import SeekAheadThread
from gslib.sig_handling import ChildProcessSignalHandler
from gslib.sig_handling import GetCaughtSignals
from gslib.sig_handling import KillProcess
from gslib.sig_handling import MultithreadedMainSignalHandler
from gslib.sig_handling import RegisterSignalHandler
from gslib.storage_url import HaveFileUrls
from gslib.storage_url import HaveProviderUrls
from gslib.storage_url import StorageUrlFromString
from gslib.storage_url import UrlsAreForSingleProvider
from gslib.storage_url import UrlsAreMixOfBucketsAndObjects
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.thread_message import FinalMessage
from gslib.thread_message import MetadataMessage
from gslib.thread_message import PerformanceSummaryMessage
from gslib.thread_message import ProducerThreadMessage
from gslib.ui_controller import MainThreadUIQueue
from gslib.ui_controller import UIController
from gslib.ui_controller import UIThread
from gslib.utils.boto_util import GetFriendlyConfigFilePaths
from gslib.utils.boto_util import GetMaxConcurrentCompressedUploads
from gslib.utils.constants import NO_MAX
from gslib.utils.constants import UTF8
import gslib.utils.parallelism_framework_util
from gslib.utils.parallelism_framework_util import AtomicDict
from gslib.utils.parallelism_framework_util import CheckMultiprocessingAvailableAndInit
from gslib.utils.parallelism_framework_util import multiprocessing_context
from gslib.utils.parallelism_framework_util import ProcessAndThreadSafeInt
from gslib.utils.parallelism_framework_util import PutToQueueWithTimeout
from gslib.utils.parallelism_framework_util import SEEK_AHEAD_JOIN_TIMEOUT
from gslib.utils.parallelism_framework_util import ShouldProhibitMultiprocessing
from gslib.utils.parallelism_framework_util import UI_THREAD_JOIN_TIMEOUT
from gslib.utils.parallelism_framework_util import ZERO_TASKS_TO_DO_ARGUMENT
from gslib.utils.rsync_util import RsyncDiffToApply
from gslib.utils.shim_util import GcloudStorageCommandMixin
from gslib.utils.system_util import GetTermLines
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.translation_helper import AclTranslation
from gslib.utils.translation_helper import GetNonMetadataHeaders
from gslib.utils.translation_helper import PRIVATE_DEFAULT_OBJ_ACL
from gslib.wildcard_iterator import CreateWildcardIterator
from six.moves import queue as Queue
def _SetAclGsutilApi(self, url, gsutil_api):
    """Sets the ACL for the URL provided using the gsutil Cloud API.

    This function assumes that self.def_acl, self.canned,
    and self.continue_on_error are initialized, and that self.acl_arg is
    either a JSON string or a canned ACL string.

    Args:
      url: CloudURL to set the ACL on.
      gsutil_api: gsutil Cloud API to use for the ACL set.
    """
    try:
        if url.IsBucket():
            if self.def_acl:
                if self.canned:
                    gsutil_api.PatchBucket(url.bucket_name, apitools_messages.Bucket(), canned_def_acl=self.acl_arg, provider=url.scheme, fields=['id'])
                else:
                    def_obj_acl = AclTranslation.JsonToMessage(self.acl_arg, apitools_messages.ObjectAccessControl)
                    if not def_obj_acl:
                        def_obj_acl.append(PRIVATE_DEFAULT_OBJ_ACL)
                    bucket_metadata = apitools_messages.Bucket(defaultObjectAcl=def_obj_acl)
                    gsutil_api.PatchBucket(url.bucket_name, bucket_metadata, provider=url.scheme, fields=['id'])
            elif self.canned:
                gsutil_api.PatchBucket(url.bucket_name, apitools_messages.Bucket(), canned_acl=self.acl_arg, provider=url.scheme, fields=['id'])
            else:
                bucket_acl = AclTranslation.JsonToMessage(self.acl_arg, apitools_messages.BucketAccessControl)
                bucket_metadata = apitools_messages.Bucket(acl=bucket_acl)
                gsutil_api.PatchBucket(url.bucket_name, bucket_metadata, provider=url.scheme, fields=['id'])
        elif self.canned:
            gsutil_api.PatchObjectMetadata(url.bucket_name, url.object_name, apitools_messages.Object(), provider=url.scheme, generation=url.generation, canned_acl=self.acl_arg)
        else:
            object_acl = AclTranslation.JsonToMessage(self.acl_arg, apitools_messages.ObjectAccessControl)
            object_metadata = apitools_messages.Object(acl=object_acl)
            gsutil_api.PatchObjectMetadata(url.bucket_name, url.object_name, object_metadata, provider=url.scheme, generation=url.generation)
    except ArgumentException as e:
        raise
    except ServiceException as e:
        if self.continue_on_error:
            self.everything_set_okay = False
            self.logger.error(e)
        else:
            raise