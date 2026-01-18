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
def _parse_mac_addr_table(self, cmd_output, mac_regex):
    """
        Parse the command output and return a dictionary which maps mac address
        to an IP address.

        :return: Dictionary which maps mac address to IP address.
        :rtype: ``dict``
        """
    lines = ensure_string(cmd_output).split('\n')
    arp_table = defaultdict(list)
    for line in lines:
        match = mac_regex.match(line)
        if not match:
            continue
        groups = match.groups()
        ip_address = groups[0]
        mac_address = groups[1]
        arp_table[mac_address].append(ip_address)
    return arp_table