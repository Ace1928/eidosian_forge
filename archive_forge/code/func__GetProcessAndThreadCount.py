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
def _GetProcessAndThreadCount(self, process_count, thread_count, parallel_operations_override, print_macos_warning=True):
    """Determines the values of process_count and thread_count.

    These values are used for parallel operations.
    If we're not performing operations in parallel, then ignore
    existing values and use process_count = thread_count = 1.

    Args:
      process_count: A positive integer or None. In the latter case, we read
                     the value from the .boto config file.
      thread_count: A positive integer or None. In the latter case, we read
                    the value from the .boto config file.
      parallel_operations_override: Used to override self.parallel_operations.
                                    This allows the caller to safely override
                                    the top-level flag for a single call.
      print_macos_warning: Print a warning about parallel processing on MacOS
                           if true.

    Returns:
      (process_count, thread_count): The number of processes and threads to use,
                                     respectively.
    """
    if self.parallel_operations or parallel_operations_override:
        if not process_count:
            process_count = boto.config.getint('GSUtil', 'parallel_process_count', gslib.commands.config.DEFAULT_PARALLEL_PROCESS_COUNT)
        if process_count < 1:
            raise CommandException('Invalid parallel_process_count "%d".' % process_count)
        if not thread_count:
            thread_count = boto.config.getint('GSUtil', 'parallel_thread_count', gslib.commands.config.DEFAULT_PARALLEL_THREAD_COUNT)
        if thread_count < 1:
            raise CommandException('Invalid parallel_thread_count "%d".' % thread_count)
    else:
        process_count = 1
        thread_count = 1
    should_prohibit_multiprocessing, os_name = ShouldProhibitMultiprocessing()
    if should_prohibit_multiprocessing and process_count > 1:
        raise CommandException('\n'.join(textwrap.wrap('It is not possible to set process_count > 1 on %s. Please update your config file(s) (located at %s) and set "parallel_process_count = 1".' % (os_name, ', '.join(GetFriendlyConfigFilePaths())))))
    is_main_thread = self.recursive_apply_level == 0
    if print_macos_warning and os_name == 'macOS' and (process_count > 1) and is_main_thread:
        self.logger.info('If you experience problems with multiprocessing on MacOS, they might be related to https://bugs.python.org/issue33725. You can disable multiprocessing by editing your .boto config or by adding the following flag to your command: `-o "GSUtil:parallel_process_count=1"`. Note that multithreading is still available even if you disable multiprocessing.\n')
    self.logger.debug('process count: %d', process_count)
    self.logger.debug('thread count: %d', thread_count)
    return (process_count, thread_count)