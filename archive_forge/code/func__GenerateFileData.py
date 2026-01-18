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
def _GenerateFileData(fp, file_size=0, random_ratio=100, max_unique_random_bytes=5242883):
    """Writes data into a file like object.

  Args:
    fp: A file like object to write the data to.
    file_size: The amount of data to write to the file.
    random_ratio: The percentage of randomly generated data to write. This can
        be any number between 0 and 100 (inclusive), with 0 producing uniform
        data, and 100 producing random data.
    max_unique_random_bytes: The maximum number of bytes to generate
                             pseudo-randomly before beginning to repeat
                             bytes. The default was chosen as the next prime
                             larger than 5 MiB.
  """
    random_ratio /= 100.0
    random_bytes = os.urandom(min(file_size, max_unique_random_bytes))
    total_bytes_written = 0
    while total_bytes_written < file_size:
        num_bytes = min(max_unique_random_bytes, file_size - total_bytes_written)
        num_bytes_seq = int(num_bytes * (1 - random_ratio))
        num_bytes_random = num_bytes - num_bytes_seq
        fp.write(random_bytes[:num_bytes_random])
        fp.write(b'x' * num_bytes_seq)
        total_bytes_written += num_bytes