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
def _SequentialApply(self, func, args_iterator, exception_handler, caller_id, arg_checker, should_return_results, fail_on_error):
    """Performs all function calls sequentially in the current thread.

    No other threads or processes will be spawned. This degraded functionality
    is used when the multiprocessing module is not available or the user
    requests only one thread and one process.
    """
    worker_thread = WorkerThread(None, False, headers=self.non_metadata_headers, perf_trace_token=self.perf_trace_token, trace_token=self.trace_token, user_project=self.user_project)
    args_iterator = iter(args_iterator)
    sequential_call_count = 0
    while True:
        try:
            args = next(args_iterator)
        except StopIteration as e:
            break
        except Exception as e:
            _IncrementFailureCount()
            if fail_on_error:
                raise
            else:
                try:
                    exception_handler(self, e)
                except Exception as _:
                    self.logger.debug('Caught exception while handling exception for %s:\n%s', func, traceback.format_exc())
                continue
        sequential_call_count += 1
        if sequential_call_count == OFFER_GSUTIL_M_SUGGESTION_THRESHOLD or sequential_call_count % OFFER_GSUTIL_M_SUGGESTION_FREQUENCY == 0:
            self._MaybeSuggestGsutilDashM()
        if arg_checker(self, args):
            task = Task(func, args, caller_id, exception_handler, should_return_results, arg_checker, fail_on_error)
            worker_thread.PerformTask(task, self)
    lines_since_suggestion_last_printed = sequential_call_count % OFFER_GSUTIL_M_SUGGESTION_FREQUENCY
    if lines_since_suggestion_last_printed >= GetTermLines():
        self._MaybeSuggestGsutilDashM()
    PutToQueueWithTimeout(self.gsutil_api.status_queue, FinalMessage(time.time()))
    worker_thread.shared_vars_updater.Update(caller_id, self)
    self._ProcessSourceUrlTypes(args_iterator)