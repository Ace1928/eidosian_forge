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
class WorkerThread(threading.Thread):
    """Thread where all the work will be performed.

  This makes the function calls for Apply and takes care of all error handling,
  return value propagation, and shared_vars.

  Note that this thread is NOT started upon instantiation because the function-
  calling logic is also used in the single-threaded case.
  """
    global thread_stats

    def __init__(self, task_queue, logger, worker_semaphore=None, bucket_storage_uri_class=None, gsutil_api_map=None, debug=0, status_queue=None, headers=None, perf_trace_token=None, trace_token=None, user_project=None):
        """Initializes the worker thread.

    Args:
      task_queue: The thread-safe queue from which this thread should obtain
                  its work.
      logger: Logger to use for this thread.
      worker_semaphore: threading.BoundedSemaphore to be released each time a
          task is completed, or None for single-threaded execution.
      bucket_storage_uri_class: Class to instantiate for cloud StorageUris.
                                Settable for testing/mocking.
      gsutil_api_map: Map of providers and API selector tuples to api classes
                      which can be used to communicate with those providers.
                      Used for the instantiating CloudApiDelegator class.
      debug: debug level for the CloudApiDelegator class.
      status_queue: Queue for reporting status updates.
      user_project: Project to be billed for this request.
    """
        super(WorkerThread, self).__init__()
        self.pid = os.getpid()
        self.init_time = time.time()
        self.task_queue = task_queue
        self.worker_semaphore = worker_semaphore
        self.daemon = True
        self.cached_classes = {}
        self.shared_vars_updater = _SharedVariablesUpdater()
        self.headers = headers
        self.perf_trace_token = perf_trace_token
        self.trace_token = trace_token
        self.user_project = user_project
        self.thread_gsutil_api = None
        if bucket_storage_uri_class and gsutil_api_map:
            self.thread_gsutil_api = CloudApiDelegator(bucket_storage_uri_class, gsutil_api_map, logger, status_queue, debug=debug, http_headers=self.headers, perf_trace_token=self.perf_trace_token, trace_token=self.trace_token, user_project=self.user_project)

    @CaptureThreadStatException
    def _StartBlockedTime(self):
        """Update the thread_stats AtomicDict before task_queue.get() is called."""
        if thread_stats.get((self.pid, self.ident)) is None:
            thread_stats[self.pid, self.ident] = _ThreadStat(self.init_time)
        thread_stat = thread_stats[self.pid, self.ident]
        thread_stat.StartBlockedTime()
        thread_stats[self.pid, self.ident] = thread_stat

    @CaptureThreadStatException
    def _EndBlockedTime(self):
        """Update the thread_stats AtomicDict after task_queue.get() is called."""
        thread_stat = thread_stats[self.pid, self.ident]
        thread_stat.EndBlockedTime()
        thread_stats[self.pid, self.ident] = thread_stat

    def PerformTask(self, task, cls):
        """Makes the function call for a task.

    Args:
      task: The Task to perform.
      cls: The instance of a class which gives context to the functions called
           by the Task's function. E.g., see SetAclFuncWrapper.
    """
        caller_id = task.caller_id
        try:
            results = task.func(cls, task.args, thread_state=self.thread_gsutil_api)
            if task.should_return_results:
                global_return_values_map.Increment(caller_id, [results], default_value=[])
        except Exception as e:
            _IncrementFailureCount()
            if task.fail_on_error:
                raise
            else:
                try:
                    task.exception_handler(cls, e)
                except Exception as _:
                    cls.logger.debug('Caught exception while handling exception for %s:\n%s', task, traceback.format_exc())
        finally:
            if self.worker_semaphore:
                self.worker_semaphore.release()
            self.shared_vars_updater.Update(caller_id, cls)
            num_done = caller_id_finished_count.Increment(caller_id, 1)
            _NotifyIfDone(caller_id, num_done)

    def run(self):
        while True:
            self._StartBlockedTime()
            task = self.task_queue.get()
            self._EndBlockedTime()
            if task.args == ZERO_TASKS_TO_DO_ARGUMENT:
                continue
            caller_id = task.caller_id
            cls = self.cached_classes.get(caller_id, None)
            if not cls:
                cls = copy.copy(class_map[caller_id])
                cls.logger = CreateOrGetGsutilLogger(cls.command_name)
                self.cached_classes[caller_id] = cls
            self.PerformTask(task, cls)