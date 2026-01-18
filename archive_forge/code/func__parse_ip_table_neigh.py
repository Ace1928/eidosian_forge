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
def _parse_ip_table_neigh(self, ip_output):
    """
        Sets up the regexp for parsing out IP addresses from the 'ip neighbor'
        command and pass it along to the parser function.

        :return: Dictionary from the parsing function
        :rtype: ``dict``
        """
    ip_regex = re.compile('(.*?)\\s+.*lladdr\\s+(.*?)\\s+')
    return self._parse_mac_addr_table(ip_output, ip_regex)