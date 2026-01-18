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
def _GetTcpStats(self):
    """Tries to parse out TCP packet information from netstat output.

    Returns:
       A dictionary containing TCP information, or None if netstat is not
       available.
    """
    try:
        netstat_output = self._Exec(['netstat', '-s'], return_output=True, raise_on_error=False)
    except OSError:
        self.logger.warning('netstat not found on your system; some measurement data will be missing')
        return None
    netstat_output = netstat_output.strip().lower()
    found_tcp = False
    tcp_retransmit = None
    tcp_received = None
    tcp_sent = None
    for line in netstat_output.split('\n'):
        if 'tcp:' in line or 'tcp statistics' in line:
            found_tcp = True
        if found_tcp and tcp_retransmit is None and ('segments retransmited' in line or 'retransmit timeouts' in line or 'segments retransmitted' in line):
            tcp_retransmit = ''.join((c for c in line if c in string.digits))
        if found_tcp and tcp_received is None and ('segments received' in line or 'packets received' in line):
            tcp_received = ''.join((c for c in line if c in string.digits))
        if found_tcp and tcp_sent is None and ('segments send out' in line or 'packets sent' in line or 'segments sent' in line):
            tcp_sent = ''.join((c for c in line if c in string.digits))
    result = {}
    try:
        result['tcp_retransmit'] = int(tcp_retransmit)
        result['tcp_received'] = int(tcp_received)
        result['tcp_sent'] = int(tcp_sent)
    except (ValueError, TypeError):
        result['tcp_retransmit'] = None
        result['tcp_received'] = None
        result['tcp_sent'] = None
    return result