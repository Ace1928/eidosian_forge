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
def _get_mac_addresses_for_domain(self, domain):
    """
        Parses network interface MAC addresses from the provided domain.
        """
    xml = domain.XMLDesc()
    etree = ET.XML(xml)
    elems = etree.findall("devices/interface[@type='network']/mac")
    result = []
    for elem in elems:
        mac_address = elem.get('address')
        result.append(mac_address)
    return result