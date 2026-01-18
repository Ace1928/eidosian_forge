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
def _SetUp(self):
    """Performs setup operations needed before diagnostics can be run."""
    self.results = {}
    self.temporary_files = set()
    self.temporary_objects = set()
    self.total_requests = 0
    self.request_errors = 0
    self.error_responses_by_code = defaultdict(int)
    self.connection_breaks = 0
    self.teardown_completed = False
    if self.LAT in self.diag_tests:
        self.latency_files = []
        for file_size in self.test_lat_file_sizes:
            fpath = self._MakeTempFile(file_size, mem_metadata=True, mem_data=True)
            self.latency_files.append(fpath)
    if self.diag_tests.intersection((self.RTHRU, self.WTHRU, self.RTHRU_FILE, self.WTHRU_FILE)):
        self.tcp_warmup_file = self._MakeTempFile(5 * 1024 * 1024, mem_metadata=True, mem_data=True)
        if self.diag_tests.intersection((self.RTHRU, self.WTHRU)):
            self.mem_thru_file_name = self._MakeTempFile(self.thru_filesize, mem_metadata=True, mem_data=True, random_ratio=self.gzip_compression_ratio)
            self.mem_thru_object_name = os.path.basename(self.mem_thru_file_name)
        if self.diag_tests.intersection((self.RTHRU_FILE, self.WTHRU_FILE)):
            self.thru_file_names = []
            self.thru_object_names = []
            free_disk_space = CheckFreeSpace(self.directory)
            if free_disk_space >= self.thru_filesize * self.num_objects:
                self.logger.info('\nCreating %d local files each of size %s.' % (self.num_objects, MakeHumanReadable(self.thru_filesize)))
                self._WarnIfLargeData()
                for _ in range(self.num_objects):
                    file_name = self._MakeTempFile(self.thru_filesize, mem_metadata=True, random_ratio=self.gzip_compression_ratio)
                    self.thru_file_names.append(file_name)
                    self.thru_object_names.append(os.path.basename(file_name))
            else:
                raise CommandException('Not enough free disk space for throughput files: %s of disk space required, but only %s available.' % (MakeHumanReadable(self.thru_filesize * self.num_objects), MakeHumanReadable(free_disk_space)))
    self.discard_sink = DummyFile()
    self.logger.addFilter(self._PerfdiagFilter())