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
def _RunWriteThruTests(self, use_file=False):
    """Runs write throughput tests."""
    test_name = 'write_throughput_file' if use_file else 'write_throughput'
    file_io_string = 'with file I/O' if use_file else ''
    self.logger.info('\nRunning write throughput tests %s (%s objects of size %s)' % (file_io_string, self.num_objects, MakeHumanReadable(self.thru_filesize)))
    self._WarnIfLargeData()
    self.results[test_name] = {'file_size': self.thru_filesize, 'processes': self.processes, 'threads': self.threads, 'parallelism': self.parallel_strategy, 'gzip_encoded_writes': self.gzip_encoded_writes, 'gzip_compression_ratio': self.gzip_compression_ratio}
    warmup_obj_name = os.path.basename(self.tcp_warmup_file)
    self.temporary_objects.add(warmup_obj_name)
    self.Upload(self.tcp_warmup_file, warmup_obj_name, self.gsutil_api)
    if use_file:
        file_names = self.thru_file_names
        object_names = self.thru_object_names
    else:
        file_names = [self.mem_thru_file_name] * self.num_objects
        object_names = [self.mem_thru_object_name + str(i) for i in range(self.num_objects)]
    for object_name in object_names:
        self.temporary_objects.add(object_name)
    t0 = time.time()
    if self.processes == 1 and self.threads == 1:
        for i in range(self.num_objects):
            self.Upload(file_names[i], object_names[i], self.gsutil_api, use_file, gzip_encoded=self.gzip_encoded_writes)
    elif self.parallel_strategy in (self.FAN, self.BOTH):
        need_to_slice = self.parallel_strategy == self.BOTH
        self.PerformFannedUpload(need_to_slice, file_names, object_names, use_file, gzip_encoded=self.gzip_encoded_writes)
    elif self.parallel_strategy == self.SLICE:
        for i in range(self.num_objects):
            self.PerformSlicedUpload(file_names[i], object_names[i], use_file, self.gsutil_api, gzip_encoded=self.gzip_encoded_writes)
    t1 = time.time()
    time_took = t1 - t0
    total_bytes_copied = self.thru_filesize * self.num_objects
    bytes_per_second = total_bytes_copied / time_took
    self.results[test_name]['time_took'] = time_took
    self.results[test_name]['total_bytes_copied'] = total_bytes_copied
    self.results[test_name]['bytes_per_second'] = bytes_per_second