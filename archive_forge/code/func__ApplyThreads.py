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
def _ApplyThreads(self, thread_count, process_count, recursive_apply_level, status_queue):
    """Assigns the work from the multi-process global task queue.

    Work is assigned to an individual process for later consumption either by
    the WorkerThreads or (if thread_count == 1) this thread.

    Args:
      thread_count: The number of threads used to perform the work. If 1, then
                    perform all work in this thread.
      process_count: The number of processes used to perform the work.
      recursive_apply_level: The depth in the tree of recursive calls to Apply
                             of this thread.
      status_queue: Multiprocessing/threading queue for progress reporting and
          performance aggregation.
    """
    assert process_count > 1, 'Invalid state, calling command._ApplyThreads with only one process.'
    _CryptoRandomAtFork()
    for catch_signal in GetCaughtSignals():
        signal.signal(catch_signal, ChildProcessSignalHandler)
    self._ResetConnectionPool()
    self.recursive_apply_level = recursive_apply_level
    task_queue = task_queues[recursive_apply_level]
    worker_semaphore = threading.BoundedSemaphore(thread_count)
    worker_pool = WorkerPool(thread_count, self.logger, worker_semaphore=worker_semaphore, bucket_storage_uri_class=self.bucket_storage_uri_class, gsutil_api_map=self.gsutil_api_map, debug=self.debug, status_queue=status_queue, headers=self.non_metadata_headers, perf_trace_token=self.perf_trace_token, trace_token=self.trace_token, user_project=self.user_project)
    num_enqueued = 0
    while True:
        while not worker_semaphore.acquire(blocking=False):
            time.sleep(0.01)
        task = task_queue.get()
        if task.args != ZERO_TASKS_TO_DO_ARGUMENT:
            worker_pool.AddTask(task)
            num_enqueued += 1
        else:
            worker_semaphore.release()