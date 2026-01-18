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
class ProducerThread(threading.Thread):
    """Thread used to enqueue work for other processes and threads."""

    def __init__(self, cls, args_iterator, caller_id, func, task_queue, should_return_results, exception_handler, arg_checker, fail_on_error, seek_ahead_iterator=None, status_queue=None):
        """Initializes the producer thread.

    Args:
      cls: Instance of Command for which this ProducerThread was created.
      args_iterator: Iterable collection of arguments to be put into the
                     work queue.
      caller_id: Globally-unique caller ID corresponding to this call to Apply.
      func: The function to be called on each element of args_iterator.
      task_queue: The queue into which tasks will be put, to later be consumed
                  by Command._ApplyThreads.
      should_return_results: True iff the results for this call to command.Apply
                             were requested.
      exception_handler: The exception handler to use when errors are
                         encountered during calls to func.
      arg_checker: Used to determine whether we should process the current
                   argument or simply skip it. Also handles any logging that
                   is specific to a particular type of argument.
      fail_on_error: If true, then raise any exceptions encountered when
                     executing func. This is only applicable in the case of
                     process_count == thread_count == 1.
      seek_ahead_iterator: If present, a seek-ahead iterator that will
          provide an approximation of the total number of tasks and bytes that
          will be iterated by the ProducerThread.
      status_queue: status_queue to inform task_queue estimation. Only
          valid when calling from the main thread, else None. Even if this is
          the main thread, the status_queue will only properly work if args
          is a collection of NameExpansionResults, which is the type that gives
          us initial information about files to be processed. Otherwise,
          nothing will be added to the queue.
    """
        super(ProducerThread, self).__init__()
        self.func = func
        self.cls = cls
        self.args_iterator = args_iterator
        self.caller_id = caller_id
        self.task_queue = task_queue
        self.arg_checker = arg_checker
        self.exception_handler = exception_handler
        self.should_return_results = should_return_results
        self.fail_on_error = fail_on_error
        self.shared_variables_updater = _SharedVariablesUpdater()
        self.daemon = True
        self.unknown_exception = None
        self.iterator_exception = None
        self.seek_ahead_iterator = seek_ahead_iterator
        self.status_queue = status_queue
        self.start()

    def run(self):
        num_tasks = 0
        cur_task = None
        last_task = None
        task_estimation_threshold = None
        seek_ahead_thread = None
        seek_ahead_thread_cancel_event = None
        seek_ahead_thread_considered = False
        args = None
        try:
            total_size = 0
            self.args_iterator = iter(self.args_iterator)
            while True:
                try:
                    args = next(self.args_iterator)
                except StopIteration as e:
                    break
                except Exception as e:
                    _IncrementFailureCount()
                    if self.fail_on_error:
                        self.iterator_exception = e
                        raise
                    else:
                        try:
                            self.exception_handler(self.cls, e)
                        except Exception as _:
                            self.cls.logger.debug('Caught exception while handling exception for %s:\n%s', self.func, traceback.format_exc())
                        self.shared_variables_updater.Update(self.caller_id, self.cls)
                        continue
                if self.arg_checker(self.cls, args):
                    num_tasks += 1
                    if self.status_queue:
                        if not num_tasks % 100:
                            if isinstance(args, NameExpansionResult) or isinstance(args, CopyObjectInfo) or isinstance(args, RsyncDiffToApply):
                                PutToQueueWithTimeout(self.status_queue, ProducerThreadMessage(num_tasks, total_size, time.time()))
                        if isinstance(args, NameExpansionResult) or isinstance(args, CopyObjectInfo):
                            if args.expanded_result:
                                json_expanded_result = json.loads(args.expanded_result)
                                if 'size' in json_expanded_result:
                                    total_size += int(json_expanded_result['size'])
                        elif isinstance(args, RsyncDiffToApply):
                            if args.copy_size:
                                total_size += int(args.copy_size)
                    if not seek_ahead_thread_considered:
                        if task_estimation_threshold is None:
                            task_estimation_threshold = _GetTaskEstimationThreshold()
                        if task_estimation_threshold <= 0:
                            seek_ahead_thread_considered = True
                        elif num_tasks >= task_estimation_threshold:
                            if self.seek_ahead_iterator:
                                seek_ahead_thread_cancel_event = threading.Event()
                                seek_ahead_thread = _StartSeekAheadThread(self.seek_ahead_iterator, seek_ahead_thread_cancel_event)
                                if boto.config.get('GSUtil', 'task_estimation_force', None):
                                    seek_ahead_thread.join(timeout=SEEK_AHEAD_JOIN_TIMEOUT)
                            seek_ahead_thread_considered = True
                    last_task = cur_task
                    cur_task = Task(self.func, args, self.caller_id, self.exception_handler, self.should_return_results, self.arg_checker, self.fail_on_error)
                    if last_task:
                        self.task_queue.put(last_task)
        except Exception as e:
            if not self.iterator_exception:
                self.unknown_exception = e
        finally:
            total_tasks[self.caller_id] = num_tasks
            if not cur_task:
                cur_task = Task(None, ZERO_TASKS_TO_DO_ARGUMENT, self.caller_id, None, None, None, None)
            self.task_queue.put(cur_task)
            if seek_ahead_thread is not None:
                seek_ahead_thread_cancel_event.set()
                seek_ahead_thread.join(timeout=SEEK_AHEAD_JOIN_TIMEOUT)
            if self.status_queue and (isinstance(args, NameExpansionResult) or isinstance(args, CopyObjectInfo) or isinstance(args, RsyncDiffToApply)):
                PutToQueueWithTimeout(self.status_queue, ProducerThreadMessage(num_tasks, total_size, time.time(), finished=True))
            _NotifyIfDone(self.caller_id, caller_id_finished_count.get(self.caller_id))