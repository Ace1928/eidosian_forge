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
def _CollectSysInfo(self):
    """Collects system information."""
    sysinfo = {}
    socket_errors = (socket.error, socket.herror, socket.gaierror, socket.timeout)
    sysinfo['boto_https_enabled'] = boto.config.get('Boto', 'is_secure', True)
    proxy_host = boto.config.get('Boto', 'proxy', None)
    proxy_port = boto.config.getint('Boto', 'proxy_port', 0)
    sysinfo['using_proxy'] = bool(proxy_host)
    if boto.config.get('Boto', 'proxy_rdns', True if proxy_host else False):
        self.logger.info('DNS lookups are disallowed in this environment, so some information is not included in this perfdiag run. To allow local DNS lookups while using a proxy, set proxy_rdns to False in your boto file.')
    try:
        sysinfo['ip_address'] = socket.gethostbyname(socket.gethostname())
    except socket_errors:
        sysinfo['ip_address'] = ''
    sysinfo['tempdir'] = self.directory
    sysinfo['gmt_timestamp'] = time.strftime('%a, %d %b %Y %H:%M:%S +0000', time.gmtime())
    cmd = ['nslookup', '-type=CNAME', self.XML_API_HOST]
    try:
        nslookup_cname_output = self._Exec(cmd, return_output=True, mute_stderr=True)
        m = re.search(' = (?P<googserv>[^.]+)\\.', nslookup_cname_output)
        sysinfo['googserv_route'] = m.group('googserv') if m else None
    except (CommandException, OSError):
        sysinfo['googserv_route'] = ''
    if IS_LINUX:
        try:
            with open('/sys/class/dmi/id/product_name', 'r') as fp:
                sysinfo['on_gce'] = fp.readline() == 'Google Compute Engine\n'
        except OSError:
            pass
        if sysinfo.get('on_gce', False):
            hostname = socket.gethostname()
            cmd = ['gcloud', 'compute', 'instances', 'list', '--filter=', hostname]
            try:
                mute_stderr = IsRunningInCiEnvironment()
                sysinfo['gce_instance_info'] = self._Exec(cmd, return_output=True, mute_stderr=mute_stderr)
            except (CommandException, OSError):
                sysinfo['gce_instance_info'] = ''
    bucket_info = self.gsutil_api.GetBucket(self.bucket_url.bucket_name, fields=['location', 'storageClass'], provider=self.bucket_url.scheme)
    sysinfo['bucket_location'] = bucket_info.location
    sysinfo['bucket_storageClass'] = bucket_info.storageClass
    try:
        t0 = time.time()
        socket.gethostbyname(self.XML_API_HOST)
        t1 = time.time()
        sysinfo['google_host_dns_latency'] = t1 - t0
    except socket_errors:
        pass
    try:
        hostname, _, ipaddrlist = socket.gethostbyname_ex(self.XML_API_HOST)
        sysinfo['googserv_ips'] = ipaddrlist
    except socket_errors:
        ipaddrlist = []
        sysinfo['googserv_ips'] = []
    sysinfo['googserv_hostnames'] = []
    for googserv_ip in ipaddrlist:
        try:
            hostname, _, ipaddrlist = socket.gethostbyaddr(googserv_ip)
            sysinfo['googserv_hostnames'].append(hostname)
        except socket_errors:
            pass
    try:
        cmd = ['nslookup', '-type=TXT', 'o-o.myaddr.google.com.']
        nslookup_txt_output = self._Exec(cmd, return_output=True, mute_stderr=True)
        m = re.search('text\\s+=\\s+"(?P<dnsip>[\\.\\d]+)"', nslookup_txt_output)
        sysinfo['dns_o-o_ip'] = m.group('dnsip') if m else None
    except (CommandException, OSError):
        sysinfo['dns_o-o_ip'] = ''
    sysinfo['google_host_connect_latencies'] = {}
    for googserv_ip in ipaddrlist:
        try:
            sock = socket.socket()
            t0 = time.time()
            sock.connect((googserv_ip, self.XML_API_PORT))
            t1 = time.time()
            sysinfo['google_host_connect_latencies'][googserv_ip] = t1 - t0
        except socket_errors:
            pass
    if proxy_host:
        proxy_ip = None
        try:
            t0 = time.time()
            proxy_ip = socket.gethostbyname(proxy_host)
            t1 = time.time()
            sysinfo['proxy_dns_latency'] = t1 - t0
        except socket_errors:
            pass
        try:
            sock = socket.socket()
            t0 = time.time()
            sock.connect((proxy_ip or proxy_host, proxy_port))
            t1 = time.time()
            sysinfo['proxy_host_connect_latency'] = t1 - t0
        except socket_errors:
            pass
    try:
        sysinfo['cpu_count'] = multiprocessing.cpu_count()
    except NotImplementedError:
        sysinfo['cpu_count'] = None
    try:
        sysinfo['load_avg'] = list(os.getloadavg())
    except (AttributeError, OSError):
        sysinfo['load_avg'] = None
    mem_total = None
    mem_free = None
    mem_buffers = None
    mem_cached = None
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    mem_total = int(''.join((c for c in line if c in string.digits))) * 1000
                elif line.startswith('MemFree'):
                    mem_free = int(''.join((c for c in line if c in string.digits))) * 1000
                elif line.startswith('Buffers'):
                    mem_buffers = int(''.join((c for c in line if c in string.digits))) * 1000
                elif line.startswith('Cached'):
                    mem_cached = int(''.join((c for c in line if c in string.digits))) * 1000
    except (IOError, ValueError):
        pass
    sysinfo['meminfo'] = {'mem_total': mem_total, 'mem_free': mem_free, 'mem_buffers': mem_buffers, 'mem_cached': mem_cached}
    sysinfo['gsutil_config'] = {}
    for attr in dir(config):
        attr_value = getattr(config, attr)
        if attr.isupper() and (not (isinstance(attr_value, six.string_types) and '\n' in attr_value)):
            sysinfo['gsutil_config'][attr] = attr_value
    sysinfo['tcp_proc_values'] = {}
    stats_to_check = ['/proc/sys/net/core/rmem_default', '/proc/sys/net/core/rmem_max', '/proc/sys/net/core/wmem_default', '/proc/sys/net/core/wmem_max', '/proc/sys/net/ipv4/tcp_timestamps', '/proc/sys/net/ipv4/tcp_sack', '/proc/sys/net/ipv4/tcp_window_scaling']
    for fname in stats_to_check:
        try:
            with open(fname, 'r') as f:
                value = f.read()
            sysinfo['tcp_proc_values'][os.path.basename(fname)] = value.strip()
        except IOError:
            pass
    self.results['sysinfo'] = sysinfo