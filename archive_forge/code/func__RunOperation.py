from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import calendar
from collections import defaultdict
from collections import namedtuple
import contextlib
import datetime
import json
import logging
import math
import multiprocessing
import os
import random
import re
import socket
import string
import subprocess
import tempfile
import time
import boto
import boto.gs.connection
import six
from six.moves import cStringIO
from six.moves import http_client
from six.moves import xrange
from six.moves import range
import gslib
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import ServiceException
from gslib.command import Command
from gslib.command import DummyArgChecker
from gslib.command_argument import CommandArgument
from gslib.commands import config
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.file_part import FilePart
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import text_util
from gslib.utils.boto_util import GetMaxRetryDelay
from gslib.utils.boto_util import ResumableThreshold
from gslib.utils.constants import UTF8
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.cloud_api_helper import GetDownloadSerializationData
from gslib.utils.hashing_helper import CalculateB64EncodedMd5FromContents
from gslib.utils.system_util import CheckFreeSpace
from gslib.utils.system_util import GetDiskCounters
from gslib.utils.system_util import GetFileSize
from gslib.utils.system_util import IS_LINUX
from gslib.utils.system_util import IsRunningInCiEnvironment
from gslib.utils.unit_util import DivideAndCeil
from gslib.utils.unit_util import HumanReadableToBytes
from gslib.utils.unit_util import MakeBitsHumanReadable
from gslib.utils.unit_util import MakeHumanReadable
from gslib.utils.unit_util import Percentile
def _RunOperation(self, func):
    """Runs an operation with retry logic.

    Args:
      func: The function to run.

    Returns:
      True if the operation succeeds, False if aborted.
    """
    success = False
    server_error_retried = 0
    total_retried = 0
    i = 0
    return_val = None
    while not success:
        next_sleep = min(random.random() * 2 ** i + 1, GetMaxRetryDelay())
        try:
            return_val = func()
            self.total_requests += 1
            success = True
        except tuple(self.exceptions) as e:
            total_retried += 1
            if total_retried > self.MAX_TOTAL_RETRIES:
                self.logger.info('Reached maximum total retries. Not retrying.')
                break
            if isinstance(e, ServiceException):
                if e.status >= 500:
                    self.error_responses_by_code[e.status] += 1
                    self.total_requests += 1
                    self.request_errors += 1
                    server_error_retried += 1
                    time.sleep(next_sleep)
                else:
                    raise
                if server_error_retried > self.MAX_SERVER_ERROR_RETRIES:
                    self.logger.info('Reached maximum server error retries. Not retrying.')
                    break
            else:
                self.connection_breaks += 1
    return return_val