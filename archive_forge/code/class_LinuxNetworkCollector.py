from __future__ import (absolute_import, division, print_function)
import glob
import os
import re
import socket
import struct
from ansible.module_utils.facts.network.base import Network, NetworkCollector
from ansible.module_utils.facts.utils import get_file_content
class LinuxNetworkCollector(NetworkCollector):
    _platform = 'Linux'
    _fact_class = LinuxNetwork
    required_facts = set(['distribution', 'platform'])