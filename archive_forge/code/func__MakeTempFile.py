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
def _MakeTempFile(self, file_size=0, mem_metadata=False, mem_data=False, prefix='gsutil_test_file', random_ratio=100):
    """Creates a temporary file of the given size and returns its path.

    Args:
      file_size: The size of the temporary file to create.
      mem_metadata: If true, store md5 and file size in memory at
                    temp_file_dict[fpath].md5, tempfile_data[fpath].file_size.
      mem_data: If true, store the file data in memory at
                temp_file_dict[fpath].data
      prefix: The prefix to use for the temporary file. Defaults to
              gsutil_test_file.
      random_ratio: The percentage of randomly generated data to write. This
          can be any number between 0 and 100 (inclusive), with 0 producing
          uniform data, and 100 producing random data.

    Returns:
      The file path of the created temporary file.
    """
    fd, fpath = tempfile.mkstemp(suffix='.bin', prefix=prefix, dir=self.directory, text=False)
    with os.fdopen(fd, 'wb') as fp:
        _GenerateFileData(fp, file_size, random_ratio)
    if mem_metadata or mem_data:
        with open(fpath, 'rb') as fp:
            file_size = GetFileSize(fp) if mem_metadata else None
            md5 = CalculateB64EncodedMd5FromContents(fp) if mem_metadata else None
            data = fp.read() if mem_data else None
            temp_file_dict[fpath] = FileDataTuple(file_size, md5, data)
    self.temporary_files.add(fpath)
    return fpath