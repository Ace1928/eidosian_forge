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
class UdpProfilesParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'allowNoPayload': 'allow_no_payload', 'bufferMaxBytes': 'buffer_max_bytes', 'bufferMaxPackets': 'buffer_max_packets', 'datagramLoadBalancing': 'datagram_load_balancing', 'defaultsFrom': 'parent', 'idleTimeout': 'idle_timeout', 'ipDfMode': 'ip_df_mode', 'ipTosToClient': 'ip_tos_to_client', 'ipTtlMode': 'ip_ttl_mode', 'ipTtlV4': 'ip_ttl_v4', 'ipTtlV6': 'ip_ttl_v6', 'linkQosToClient': 'link_qos_to_client', 'noChecksum': 'no_checksum', 'proxyMss': 'proxy_mss'}
    returnables = ['full_path', 'name', 'parent', 'description', 'allow_no_payload', 'buffer_max_bytes', 'buffer_max_packets', 'datagram_load_balancing', 'idle_timeout', 'ip_df_mode', 'ip_tos_to_client', 'ip_ttl_mode', 'ip_ttl_v4', 'ip_ttl_v6', 'link_qos_to_client', 'no_checksum', 'proxy_mss']

    @property
    def description(self):
        if self._values['description'] in [None, 'none']:
            return None
        return self._values['description']

    @property
    def allow_no_payload(self):
        return flatten_boolean(self._values['allow_no_payload'])

    @property
    def datagram_load_balancing(self):
        return flatten_boolean(self._values['datagram_load_balancing'])

    @property
    def proxy_mss(self):
        return flatten_boolean(self._values['proxy_mss'])

    @property
    def no_checksum(self):
        return flatten_boolean(self._values['no_checksum'])