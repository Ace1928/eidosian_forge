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
def _DisplayResults(self):
    """Displays results collected from diagnostic run."""
    text_util.print_to_fd()
    text_util.print_to_fd('=' * 78)
    text_util.print_to_fd('DIAGNOSTIC RESULTS'.center(78))
    text_util.print_to_fd('=' * 78)
    if 'latency' in self.results:
        text_util.print_to_fd()
        text_util.print_to_fd('-' * 78)
        text_util.print_to_fd('Latency'.center(78))
        text_util.print_to_fd('-' * 78)
        text_util.print_to_fd('Operation       Size  Trials  Mean (ms)  Std Dev (ms)  Median (ms)  90th % (ms)')
        text_util.print_to_fd('=========  =========  ======  =========  ============  ===========  ===========')
        for key in sorted(self.results['latency']):
            trials = sorted(self.results['latency'][key])
            op, numbytes = key.split('_')
            numbytes = int(numbytes)
            if op == 'METADATA':
                text_util.print_to_fd('Metadata'.rjust(9), '', end=' ')
                text_util.print_to_fd(MakeHumanReadable(numbytes).rjust(9), '', end=' ')
                self._DisplayStats(trials)
            if op == 'DOWNLOAD':
                text_util.print_to_fd('Download'.rjust(9), '', end=' ')
                text_util.print_to_fd(MakeHumanReadable(numbytes).rjust(9), '', end=' ')
                self._DisplayStats(trials)
            if op == 'UPLOAD':
                text_util.print_to_fd('Upload'.rjust(9), '', end=' ')
                text_util.print_to_fd(MakeHumanReadable(numbytes).rjust(9), '', end=' ')
                self._DisplayStats(trials)
            if op == 'DELETE':
                text_util.print_to_fd('Delete'.rjust(9), '', end=' ')
                text_util.print_to_fd(MakeHumanReadable(numbytes).rjust(9), '', end=' ')
                self._DisplayStats(trials)
    if 'write_throughput' in self.results:
        text_util.print_to_fd()
        text_util.print_to_fd('-' * 78)
        text_util.print_to_fd('Write Throughput'.center(78))
        text_util.print_to_fd('-' * 78)
        write_thru = self.results['write_throughput']
        text_util.print_to_fd('Copied %s %s file(s) for a total transfer size of %s.' % (self.num_objects, MakeHumanReadable(write_thru['file_size']), MakeHumanReadable(write_thru['total_bytes_copied'])))
        text_util.print_to_fd('Write throughput: %s/s.' % MakeBitsHumanReadable(write_thru['bytes_per_second'] * 8))
        if 'parallelism' in write_thru:
            text_util.print_to_fd('Parallelism strategy: %s' % write_thru['parallelism'])
    if 'write_throughput_file' in self.results:
        text_util.print_to_fd()
        text_util.print_to_fd('-' * 78)
        text_util.print_to_fd('Write Throughput With File I/O'.center(78))
        text_util.print_to_fd('-' * 78)
        write_thru_file = self.results['write_throughput_file']
        text_util.print_to_fd('Copied %s %s file(s) for a total transfer size of %s.' % (self.num_objects, MakeHumanReadable(write_thru_file['file_size']), MakeHumanReadable(write_thru_file['total_bytes_copied'])))
        text_util.print_to_fd('Write throughput: %s/s.' % MakeBitsHumanReadable(write_thru_file['bytes_per_second'] * 8))
        if 'parallelism' in write_thru_file:
            text_util.print_to_fd('Parallelism strategy: %s' % write_thru_file['parallelism'])
    if 'read_throughput' in self.results:
        text_util.print_to_fd()
        text_util.print_to_fd('-' * 78)
        text_util.print_to_fd('Read Throughput'.center(78))
        text_util.print_to_fd('-' * 78)
        read_thru = self.results['read_throughput']
        text_util.print_to_fd('Copied %s %s file(s) for a total transfer size of %s.' % (self.num_objects, MakeHumanReadable(read_thru['file_size']), MakeHumanReadable(read_thru['total_bytes_copied'])))
        text_util.print_to_fd('Read throughput: %s/s.' % MakeBitsHumanReadable(read_thru['bytes_per_second'] * 8))
        if 'parallelism' in read_thru:
            text_util.print_to_fd('Parallelism strategy: %s' % read_thru['parallelism'])
    if 'read_throughput_file' in self.results:
        text_util.print_to_fd()
        text_util.print_to_fd('-' * 78)
        text_util.print_to_fd('Read Throughput With File I/O'.center(78))
        text_util.print_to_fd('-' * 78)
        read_thru_file = self.results['read_throughput_file']
        text_util.print_to_fd('Copied %s %s file(s) for a total transfer size of %s.' % (self.num_objects, MakeHumanReadable(read_thru_file['file_size']), MakeHumanReadable(read_thru_file['total_bytes_copied'])))
        text_util.print_to_fd('Read throughput: %s/s.' % MakeBitsHumanReadable(read_thru_file['bytes_per_second'] * 8))
        if 'parallelism' in read_thru_file:
            text_util.print_to_fd('Parallelism strategy: %s' % read_thru_file['parallelism'])
    if 'listing' in self.results:
        text_util.print_to_fd()
        text_util.print_to_fd('-' * 78)
        text_util.print_to_fd('Listing'.center(78))
        text_util.print_to_fd('-' * 78)
        listing = self.results['listing']
        insert = listing['insert']
        delete = listing['delete']
        text_util.print_to_fd('After inserting %s objects:' % listing['num_files'])
        text_util.print_to_fd('  Total time for objects to appear: %.2g seconds' % insert['time_took'])
        text_util.print_to_fd('  Number of listing calls made: %s' % insert['num_listing_calls'])
        text_util.print_to_fd('  Individual listing call latencies: [%s]' % ', '.join(('%.2gs' % lat for lat in insert['list_latencies'])))
        text_util.print_to_fd('  Files reflected after each call: [%s]' % ', '.join(map(str, insert['files_seen_after_listing'])))
        text_util.print_to_fd('After deleting %s objects:' % listing['num_files'])
        text_util.print_to_fd('  Total time for objects to appear: %.2g seconds' % delete['time_took'])
        text_util.print_to_fd('  Number of listing calls made: %s' % delete['num_listing_calls'])
        text_util.print_to_fd('  Individual listing call latencies: [%s]' % ', '.join(('%.2gs' % lat for lat in delete['list_latencies'])))
        text_util.print_to_fd('  Files reflected after each call: [%s]' % ', '.join(map(str, delete['files_seen_after_listing'])))
    if 'sysinfo' in self.results:
        text_util.print_to_fd()
        text_util.print_to_fd('-' * 78)
        text_util.print_to_fd('System Information'.center(78))
        text_util.print_to_fd('-' * 78)
        info = self.results['sysinfo']
        text_util.print_to_fd('IP Address: \n  %s' % info['ip_address'])
        text_util.print_to_fd('Temporary Directory: \n  %s' % info['tempdir'])
        text_util.print_to_fd('Bucket URI: \n  %s' % self.results['bucket_uri'])
        text_util.print_to_fd('gsutil Version: \n  %s' % self.results.get('gsutil_version', 'Unknown'))
        text_util.print_to_fd('boto Version: \n  %s' % self.results.get('boto_version', 'Unknown'))
        if 'gmt_timestamp' in info:
            ts_string = info['gmt_timestamp']
            timetuple = None
            try:
                timetuple = time.strptime(ts_string, '%a, %d %b %Y %H:%M:%S +0000')
            except ValueError:
                pass
            if timetuple:
                localtime = calendar.timegm(timetuple)
                localdt = datetime.datetime.fromtimestamp(localtime)
                text_util.print_to_fd('Measurement time: \n %s' % localdt.strftime('%Y-%m-%d %I:%M:%S %p %Z'))
        if 'on_gce' in info:
            text_util.print_to_fd('Running on GCE: \n  %s' % info['on_gce'])
            if info['on_gce']:
                text_util.print_to_fd('GCE Instance:\n\t%s' % info['gce_instance_info'].replace('\n', '\n\t'))
        text_util.print_to_fd('Bucket location: \n  %s' % info['bucket_location'])
        text_util.print_to_fd('Bucket storage class: \n  %s' % info['bucket_storageClass'])
        text_util.print_to_fd('Google Server: \n  %s' % info['googserv_route'])
        text_util.print_to_fd('Google Server IP Addresses: \n  %s' % '\n  '.join(info['googserv_ips']))
        text_util.print_to_fd('Google Server Hostnames: \n  %s' % '\n  '.join(info['googserv_hostnames']))
        text_util.print_to_fd('Google DNS thinks your IP is: \n  %s' % info['dns_o-o_ip'])
        text_util.print_to_fd('CPU Count: \n  %s' % info['cpu_count'])
        text_util.print_to_fd('CPU Load Average: \n  %s' % info['load_avg'])
        try:
            text_util.print_to_fd('Total Memory: \n  %s' % MakeHumanReadable(info['meminfo']['mem_total']))
            text_util.print_to_fd('Free Memory: \n  %s' % MakeHumanReadable(info['meminfo']['mem_free'] + info['meminfo']['mem_buffers'] + info['meminfo']['mem_cached']))
        except TypeError:
            pass
        if 'netstat_end' in info and 'netstat_start' in info:
            netstat_after = info['netstat_end']
            netstat_before = info['netstat_start']
            for tcp_type in ('sent', 'received', 'retransmit'):
                try:
                    delta = netstat_after['tcp_%s' % tcp_type] - netstat_before['tcp_%s' % tcp_type]
                    text_util.print_to_fd('TCP segments %s during test:\n  %d' % (tcp_type, delta))
                except TypeError:
                    pass
        else:
            text_util.print_to_fd('TCP segment counts not available because "netstat" was not found during test runs')
        if 'disk_counters_end' in info and 'disk_counters_start' in info:
            text_util.print_to_fd('Disk Counter Deltas:\n', end=' ')
            disk_after = info['disk_counters_end']
            disk_before = info['disk_counters_start']
            text_util.print_to_fd('', 'disk'.rjust(6), end=' ')
            for colname in ['reads', 'writes', 'rbytes', 'wbytes', 'rtime', 'wtime']:
                text_util.print_to_fd(colname.rjust(8), end=' ')
            text_util.print_to_fd()
            for diskname in sorted(disk_after):
                before = disk_before[diskname]
                after = disk_after[diskname]
                reads1, writes1, rbytes1, wbytes1, rtime1, wtime1 = before
                reads2, writes2, rbytes2, wbytes2, rtime2, wtime2 = after
                text_util.print_to_fd('', diskname.rjust(6), end=' ')
                deltas = [reads2 - reads1, writes2 - writes1, rbytes2 - rbytes1, wbytes2 - wbytes1, rtime2 - rtime1, wtime2 - wtime1]
                for delta in deltas:
                    text_util.print_to_fd(str(delta).rjust(8), end=' ')
                text_util.print_to_fd()
        if 'tcp_proc_values' in info:
            text_util.print_to_fd('TCP /proc values:\n', end=' ')
            for item in six.iteritems(info['tcp_proc_values']):
                text_util.print_to_fd('   %s = %s' % item)
        if 'boto_https_enabled' in info:
            text_util.print_to_fd('Boto HTTPS Enabled: \n  %s' % info['boto_https_enabled'])
        if 'using_proxy' in info:
            text_util.print_to_fd('Requests routed through proxy: \n  %s' % info['using_proxy'])
        if 'google_host_dns_latency' in info:
            text_util.print_to_fd('Latency of the DNS lookup for Google Storage server (ms): \n  %.1f' % (info['google_host_dns_latency'] * 1000.0))
        if 'google_host_connect_latencies' in info:
            text_util.print_to_fd('Latencies connecting to Google Storage server IPs (ms):')
            for ip, latency in six.iteritems(info['google_host_connect_latencies']):
                text_util.print_to_fd('  %s = %.1f' % (ip, latency * 1000.0))
        if 'proxy_dns_latency' in info:
            text_util.print_to_fd('Latency of the DNS lookup for the configured proxy (ms): \n  %.1f' % (info['proxy_dns_latency'] * 1000.0))
        if 'proxy_host_connect_latency' in info:
            text_util.print_to_fd('Latency connecting to the configured proxy (ms): \n  %.1f' % (info['proxy_host_connect_latency'] * 1000.0))
    if 'request_errors' in self.results and 'total_requests' in self.results:
        text_util.print_to_fd()
        text_util.print_to_fd('-' * 78)
        text_util.print_to_fd('In-Process HTTP Statistics'.center(78))
        text_util.print_to_fd('-' * 78)
        total = int(self.results['total_requests'])
        numerrors = int(self.results['request_errors'])
        numbreaks = int(self.results['connection_breaks'])
        availability = (total - numerrors) / float(total) * 100 if total > 0 else 100
        text_util.print_to_fd('Total HTTP requests made: %d' % total)
        text_util.print_to_fd('HTTP 5xx errors: %d' % numerrors)
        text_util.print_to_fd('HTTP connections broken: %d' % numbreaks)
        text_util.print_to_fd('Availability: %.7g%%' % availability)
        if 'error_responses_by_code' in self.results:
            sorted_codes = sorted(six.iteritems(self.results['error_responses_by_code']))
            if sorted_codes:
                text_util.print_to_fd('Error responses by code:')
                text_util.print_to_fd('\n'.join(('  %s: %s' % c for c in sorted_codes)))
    if self.output_file:
        with open(self.output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        text_util.print_to_fd()
        text_util.print_to_fd("Output file written to '%s'." % self.output_file)
    text_util.print_to_fd()