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
def PerformFannedUpload(self, need_to_slice, file_names, object_names, use_file, gzip_encoded=False):
    """Performs a parallel upload of multiple files using the fan strategy.

    The metadata for file_name should be present in temp_file_dict prior
    to calling. Also, the data for file_name should be present in temp_file_dict
    if use_file is specified as False.

    Args:
      need_to_slice: If True, additionally apply the slice strategy to each
                     file in file_names.
      file_names: A list of file names to be uploaded.
      object_names: A list, corresponding by by index to file_names, of object
                    names to upload data to.
      use_file: If true, use disk I/O, otherwise read upload data from memory.
      gzip_encoded: Flag for if the file will be uploaded with the gzip
                    transport encoding. If true, a lock is used to limit
                    resource usage.
    """
    args = []
    for i in range(len(file_names)):
        args.append(FanUploadTuple(need_to_slice, file_names[i], object_names[i], use_file, gzip_encoded))
    self.Apply(_UploadObject, args, _PerfdiagExceptionHandler, ('total_requests', 'request_errors'), arg_checker=DummyArgChecker, parallel_operations_override=self.ParallelOverrideReason.PERFDIAG, process_count=self.processes, thread_count=self.threads)