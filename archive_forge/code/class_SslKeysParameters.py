from __future__ import absolute_import, division, print_function
import datetime
import math
import re
import time
import traceback
from collections import namedtuple
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE
from ansible.module_utils.six import (
from ansible.module_utils.urls import urlparse
from ipaddress import ip_interface
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.urls import parseStats
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
class SslKeysParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'keyType': 'key_type', 'keySize': 'key_size', 'securityType': 'security_type', 'systemPath': 'system_path', 'checksum': 'sha1_checksum'}
    returnables = ['full_path', 'name', 'key_type', 'key_size', 'security_type', 'system_path', 'sha1_checksum']

    @property
    def sha1_checksum(self):
        if self._values['sha1_checksum'] is None:
            return None
        parts = self._values['sha1_checksum'].split(':')
        return parts[2]