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
class FastL4ProfilesParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'clientTimeout': 'client_timeout', 'defaultsFrom': 'parent', 'explicitFlowMigration': 'explicit_flow_migration', 'hardwareSynCookie': 'hardware_syn_cookie', 'idleTimeout': 'idle_timeout', 'ipDfMode': 'dont_fragment_flag', 'ipTosToClient': 'ip_tos_to_client', 'ipTosToServer': 'ip_tos_to_server', 'ipTtlMode': 'ttl_mode', 'ipTtlV4': 'ttl_v4', 'ipTtlV6': 'ttl_v6', 'keepAliveInterval': 'keep_alive_interval', 'lateBinding': 'late_binding', 'linkQosToClient': 'link_qos_to_client', 'linkQosToServer': 'link_qos_to_server', 'looseClose': 'loose_close', 'looseInitialization': 'loose_init', 'mssOverride': 'mss_override', 'priorityToClient': 'priority_to_client', 'priorityToServer': 'priority_to_server', 'pvaAcceleration': 'pva_acceleration', 'pvaDynamicClientPackets': 'pva_dynamic_client_packets', 'pvaDynamicServerPackets': 'pva_dynamic_server_packets', 'pvaFlowAging': 'pva_flow_aging', 'pvaFlowEvict': 'pva_flow_evict', 'pvaOffloadDynamic': 'pva_offload_dynamic', 'pvaOffloadState': 'pva_offload_state', 'reassembleFragments': 'reassemble_fragments', 'receiveWindowSize': 'receive_window', 'resetOnTimeout': 'reset_on_timeout', 'rttFromClient': 'rtt_from_client', 'rttFromServer': 'rtt_from_server', 'serverSack': 'server_sack', 'serverTimestamp': 'server_timestamp', 'softwareSynCookie': 'software_syn_cookie', 'synCookieEnable': 'syn_cookie_enabled', 'synCookieMss': 'syn_cookie_mss', 'synCookieWhitelist': 'syn_cookie_whitelist', 'tcpCloseTimeout': 'tcp_close_timeout', 'tcpGenerateIsn': 'generate_init_seq_number', 'tcpHandshakeTimeout': 'tcp_handshake_timeout', 'tcpStripSack': 'strip_sack', 'tcpTimeWaitTimeout': 'tcp_time_wait_timeout', 'tcpTimestampMode': 'tcp_timestamp_mode', 'tcpWscaleMode': 'tcp_window_scale_mode', 'timeoutRecovery': 'timeout_recovery'}
    returnables = ['full_path', 'name', 'client_timeout', 'parent', 'description', 'explicit_flow_migration', 'hardware_syn_cookie', 'idle_timeout', 'dont_fragment_flag', 'ip_tos_to_client', 'ip_tos_to_server', 'ttl_mode', 'ttl_v4', 'ttl_v6', 'keep_alive_interval', 'late_binding', 'link_qos_to_client', 'link_qos_to_server', 'loose_close', 'loose_init', 'mss_override', 'priority_to_client', 'priority_to_server', 'pva_acceleration', 'pva_dynamic_client_packets', 'pva_dynamic_server_packets', 'pva_flow_aging', 'pva_flow_evict', 'pva_offload_dynamic', 'pva_offload_state', 'reassemble_fragments', 'receive_window', 'reset_on_timeout', 'rtt_from_client', 'rtt_from_server', 'server_sack', 'server_timestamp', 'software_syn_cookie', 'syn_cookie_enabled', 'syn_cookie_mss', 'syn_cookie_whitelist', 'tcp_close_timeout', 'generate_init_seq_number', 'tcp_handshake_timeout', 'strip_sack', 'tcp_time_wait_timeout', 'tcp_timestamp_mode', 'tcp_window_scale_mode', 'timeout_recovery']

    @property
    def description(self):
        if self._values['description'] in [None, 'none']:
            return None
        return self._values['description']

    @property
    def strip_sack(self):
        return flatten_boolean(self._values['strip_sack'])

    @property
    def generate_init_seq_number(self):
        return flatten_boolean(self._values['generate_init_seq_number'])

    @property
    def syn_cookie_whitelist(self):
        return flatten_boolean(self._values['syn_cookie_whitelist'])

    @property
    def syn_cookie_enabled(self):
        return flatten_boolean(self._values['syn_cookie_enabled'])

    @property
    def software_syn_cookie(self):
        return flatten_boolean(self._values['software_syn_cookie'])

    @property
    def server_timestamp(self):
        return flatten_boolean(self._values['server_timestamp'])

    @property
    def server_sack(self):
        return flatten_boolean(self._values['server_sack'])

    @property
    def rtt_from_server(self):
        return flatten_boolean(self._values['rtt_from_server'])

    @property
    def rtt_from_client(self):
        return flatten_boolean(self._values['rtt_from_client'])

    @property
    def reset_on_timeout(self):
        return flatten_boolean(self._values['reset_on_timeout'])

    @property
    def explicit_flow_migration(self):
        return flatten_boolean(self._values['explicit_flow_migration'])

    @property
    def reassemble_fragments(self):
        return flatten_boolean(self._values['reassemble_fragments'])

    @property
    def pva_flow_aging(self):
        return flatten_boolean(self._values['pva_flow_aging'])

    @property
    def pva_flow_evict(self):
        return flatten_boolean(self._values['pva_flow_evict'])

    @property
    def pva_offload_dynamic(self):
        return flatten_boolean(self._values['pva_offload_dynamic'])

    @property
    def hardware_syn_cookie(self):
        return flatten_boolean(self._values['hardware_syn_cookie'])

    @property
    def loose_close(self):
        return flatten_boolean(self._values['loose_close'])

    @property
    def loose_init(self):
        return flatten_boolean(self._values['loose_init'])

    @property
    def late_binding(self):
        return flatten_boolean(self._values['late_binding'])

    @property
    def tcp_handshake_timeout(self):
        if self._values['tcp_handshake_timeout'] is None:
            return None
        elif self._values['tcp_handshake_timeout'] == 'immediate':
            return 0
        elif self._values['tcp_handshake_timeout'] == 'indefinite':
            return 4294967295
        return int(self._values['tcp_handshake_timeout'])

    @property
    def idle_timeout(self):
        if self._values['idle_timeout'] is None:
            return None
        elif self._values['idle_timeout'] == 'immediate':
            return 0
        elif self._values['idle_timeout'] == 'indefinite':
            return 4294967295
        return int(self._values['idle_timeout'])

    @property
    def tcp_close_timeout(self):
        if self._values['tcp_close_timeout'] is None:
            return None
        elif self._values['tcp_close_timeout'] == 'immediate':
            return 0
        elif self._values['tcp_close_timeout'] == 'indefinite':
            return 4294967295
        return int(self._values['tcp_close_timeout'])

    @property
    def keep_alive_interval(self):
        if self._values['keep_alive_interval'] is None:
            return None
        elif self._values['keep_alive_interval'] == 'disabled':
            return 0
        return int(self._values['keep_alive_interval'])

    @property
    def ip_tos_to_client(self):
        if self._values['ip_tos_to_client'] is None:
            return None
        try:
            return int(self._values['ip_tos_to_client'])
        except ValueError:
            return self._values['ip_tos_to_client']

    @property
    def ip_tos_to_server(self):
        if self._values['ip_tos_to_server'] is None:
            return None
        try:
            return int(self._values['ip_tos_to_server'])
        except ValueError:
            return self._values['ip_tos_to_server']

    @property
    def link_qos_to_client(self):
        if self._values['link_qos_to_client'] is None:
            return None
        try:
            return int(self._values['link_qos_to_client'])
        except ValueError:
            return self._values['link_qos_to_client']

    @property
    def link_qos_to_server(self):
        if self._values['link_qos_to_server'] is None:
            return None
        try:
            return int(self._values['link_qos_to_server'])
        except ValueError:
            return self._values['link_qos_to_server']

    @property
    def priority_to_client(self):
        if self._values['priority_to_client'] is None:
            return None
        try:
            return int(self._values['priority_to_client'])
        except ValueError:
            return self._values['priority_to_client']

    @property
    def priority_to_server(self):
        if self._values['priority_to_server'] is None:
            return None
        try:
            return int(self._values['priority_to_server'])
        except ValueError:
            return self._values['priority_to_server']