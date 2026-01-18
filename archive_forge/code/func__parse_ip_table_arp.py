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
def _parse_ip_table_arp(self, arp_output):
    """
        Sets up the regexp for parsing out IP addresses from the 'arp -an'
        command and pass it along to the parser function.

        :return: Dictionary from the parsing function
        :rtype: ``dict``
        """
    arp_regex = re.compile('.*?\\((.*?)\\) at (.*?)\\s+')
    return self._parse_mac_addr_table(arp_output, arp_regex)