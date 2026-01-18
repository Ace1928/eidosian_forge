from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import sys
import threading
import time
from apitools.base.py import encoding_helper
from apitools.base.py.exceptions import HttpConflictError
from apitools.base.py.exceptions import HttpError
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import iap_tunnel
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import exceptions as tpu_exceptions
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import util as tpu_utils
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util.files import FileWriter
import six
def ParseWorkerFlag(worker_flag, network_endpoints, use_internal_ips):
    """Parses the --worker flag into a dict of worker indexes to IP addresses."""
    n_vms = len(network_endpoints)
    if six.text_type(worker_flag).upper() == 'ALL':
        indexes = list(range(n_vms))
    else:
        indexes = set()
        ranges = worker_flag.split(',')
        for r in ranges:
            if not r:
                continue
            if '-' in r:
                bounds = r.split('-')
                if len(bounds) != 2 or not bounds[0] or (not bounds[1]):
                    raise exceptions.InvalidArgumentException('--worker', 'found malformed range "{}".'.format(r))
                start, end = (int(bounds[0]), int(bounds[1]))
                if start >= end:
                    raise exceptions.InvalidArgumentException('--worker', 'found malformed range "{}".'.format(r))
                indexes.update(range(start, end + 1))
            else:
                try:
                    indexes.add(int(r))
                except ValueError:
                    raise exceptions.InvalidArgumentException('--worker', 'unable to parse worker ID {}. Please only usenumbers.'.format(r))
    if not indexes:
        raise exceptions.InvalidArgumentException('--worker', 'no worker specified, or none were parsed from the argument value.')
    mx = max(indexes)
    if mx >= n_vms:
        raise exceptions.InvalidArgumentException('--worker', 'worker index {} is larger than the valid worker indexes on this TPU VM. Please only use indexes in the range [0, {}], inclusive.'.format(mx, n_vms - 1))
    worker_ips = {}
    for worker in indexes:
        internal_address = network_endpoints[worker].ipAddress
        if not use_internal_ips and network_endpoints[worker].accessConfig and network_endpoints[worker].accessConfig.externalIp:
            ip_address = network_endpoints[worker].accessConfig.externalIp
        else:
            ip_address = internal_address
        worker_ips[worker] = IPAddresses(ip_address, internal_address)
    return worker_ips