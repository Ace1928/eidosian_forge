import os
import re
import time
import platform
import mimetypes
import subprocess
from os.path import join as pjoin
from collections import defaultdict
from libcloud.utils.py3 import ET, ensure_string
from libcloud.compute.base import Node, NodeState, NodeDriver
from libcloud.compute.types import Provider
from libcloud.utils.networking import is_public_subnet
def _get_ip_addresses_for_domain(self, domain):
    """
        Retrieve IP addresses for the provided domain.

        Note: This functionality is currently only supported on Linux and
        only works if this code is run on the same machine as the VMs run
        on.

        :return: IP addresses for the provided domain.
        :rtype: ``list``
        """
    result = []
    if platform.system() != 'Linux':
        return result
    if '///' not in self._uri:
        return result
    mac_addresses = self._get_mac_addresses_for_domain(domain=domain)
    arp_table = {}
    try:
        cmd = ['arp', '-an']
        child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = child.communicate()
        arp_table = self._parse_ip_table_arp(arp_output=stdout)
    except OSError as e:
        if e.errno == 2:
            cmd = ['ip', 'neigh']
            child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, _ = child.communicate()
            arp_table = self._parse_ip_table_neigh(ip_output=stdout)
    for mac_address in mac_addresses:
        if mac_address in arp_table:
            ip_addresses = arp_table[mac_address]
            result.extend(ip_addresses)
    return result