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
def SetAclCommandHelper(self, acl_func, acl_excep_handler):
    """Sets ACLs on the self.args using the passed-in acl function.

    Args:
      acl_func: ACL function to be passed to Apply.
      acl_excep_handler: ACL exception handler to be passed to Apply.
    """
    acl_arg = self.args[0]
    url_args = self.args[1:]
    if not UrlsAreForSingleProvider(url_args):
        raise CommandException('"%s" command spanning providers not allowed.' % self.command_name)
    if os.path.isfile(acl_arg):
        with codecs.open(acl_arg, 'r', UTF8) as f:
            acl_arg = f.read()
        self.canned = False
    else:
        storage_uri = boto.storage_uri(url_args[0], debug=self.debug, validate=False, bucket_storage_uri_class=self.bucket_storage_uri_class)
        canned_acls = storage_uri.canned_acls()
        if acl_arg not in canned_acls:
            raise CommandException('Invalid canned ACL "%s".' % acl_arg)
        self.canned = True
    self.everything_set_okay = True
    self.acl_arg = acl_arg
    self.ApplyAclFunc(acl_func, acl_excep_handler, url_args)
    if not self.everything_set_okay and (not self.continue_on_error):
        raise CommandException('ACLs for some objects could not be set.')