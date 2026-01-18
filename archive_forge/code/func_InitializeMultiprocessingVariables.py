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
def InitializeMultiprocessingVariables():
    """Initializes module-level variables that will be inherited by subprocesses.

  On Windows, a multiprocessing.Manager object should only
  be created within an "if __name__ == '__main__':" block. This function
  must be called, otherwise every command that calls Command.Apply will fail.

  While multiprocessing variables are initialized at the beginning of
  gsutil execution, new processes and threads are created only by calls
  to Command.Apply. When multiple processes and threads are used,
  the flow of startup/teardown looks like this:

  1. __main__: initializes multiprocessing variables, including any necessary
     Manager processes (here and in gslib.utils.parallelism_framework_util).
  2. __main__: Registers signal handlers for terminating signals responsible
     for cleaning up multiprocessing variables and manager processes upon exit.
  3. Command.Apply registers signal handlers for the main process to kill
     itself after the cleanup handlers registered by __main__ have executed.
  4. If worker processes have not been created for the current level of
     recursive calls, Command.Apply creates those processes.

  ---- Parallel operations start here, so steps are no longer numbered. ----
  - Command.Apply in the main thread starts the ProducerThread.
    - The Producer thread adds task arguments to the global task queue.
      - It optionally starts the SeekAheadThread which estimates total
        work for the Apply call.

  - Command.Apply in the main thread starts the UIThread, which will consume
    messages from the global status queue, process them, and display them to
    the user.

  - Each worker process creates a thread pool to perform work.
    - The worker process registers signal handlers to kill itself in
      response to a terminating signal.
    - The main thread of the worker process moves items from the global
      task queue to the process-local task queue.
    - Worker threads retrieve items from the process-local task queue,
      perform the work, and post messages to the global status queue.
    - Worker threads may themselves call Command.Apply.
      - This creates a new pool of worker subprocesses with the same size
        as the main pool. This pool is shared amongst all Command.Apply calls
        at the given recursion depth.
      - This reuses the global UIThread, global status queue, and global task
        queue.
      - This starts a new ProducerThread.
      - A SeekAheadThread is not started at this level; only one such thread
        exists at the top level, and it provides estimates for top-level work
        only.

  - The ProducerThread runs out of tasks, or the user signals cancellation.
    - The ProducerThread cancels the SeekAheadThread (if it is running) via
      an event.
    - The ProducerThread enqueues special terminating messages on the
      global task queue and global status queue, signaling the UI Thread to
      shut down and the main thread to continue operation.
    - In the termination case, existing processes exit in response to
      terminating signals from the main process.

  ---- Parallel operations end here. ----
  5. Further top-level calls to Command.Apply can be made, which will repeat
     all of the steps made in #4, except that worker processes will be
     reused.
  """
    global manager, consumer_pools, task_queues, caller_id_lock, caller_id_counter
    global total_tasks, call_completed_map, global_return_values_map, thread_stats
    global need_pool_or_done_cond, caller_id_finished_count, new_pool_needed
    global current_max_recursive_level, shared_vars_map, shared_vars_list_map
    global class_map, worker_checking_level_lock, failure_count, glob_status_queue
    global concurrent_compressed_upload_lock
    manager = multiprocessing_context.Manager()
    consumer_pools = []
    task_queues = []
    caller_id_lock = manager.Lock()
    caller_id_counter = ProcessAndThreadSafeInt(True)
    total_tasks = AtomicDict(manager=manager)
    call_completed_map = AtomicDict(manager=manager)
    global_return_values_map = AtomicDict(manager=manager)
    need_pool_or_done_cond = manager.Condition()
    worker_checking_level_lock = manager.Lock()
    caller_id_finished_count = AtomicDict(manager=manager)
    new_pool_needed = ProcessAndThreadSafeInt(True)
    current_max_recursive_level = ProcessAndThreadSafeInt(True)
    shared_vars_map = AtomicDict(manager=manager)
    shared_vars_list_map = AtomicDict(manager=manager)
    thread_stats = AtomicDict(manager=manager)
    class_map = manager.dict()
    failure_count = ProcessAndThreadSafeInt(True)
    glob_status_queue = manager.Queue(MAX_QUEUE_SIZE)
    concurrent_compressed_upload_lock = manager.BoundedSemaphore(GetMaxConcurrentCompressedUploads())