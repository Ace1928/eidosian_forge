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
class FastHttpProfilesParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'clientCloseTimeout': 'client_close_timeout', 'connpoolIdleTimeoutOverride': 'oneconnect_idle_timeout_override', 'connpoolMaxReuse': 'oneconnect_maximum_reuse', 'connpoolMaxSize': 'oneconnect_maximum_pool_size', 'connpoolMinSize': 'oneconnect_minimum_pool_size', 'connpoolReplenish': 'oneconnect_replenish', 'connpoolStep': 'oneconnect_ramp_up_increment', 'defaultsFrom': 'parent', 'forceHttp_10Response': 'force_http_1_0_response', 'headerInsert': 'request_header_insert', 'http_11CloseWorkarounds': 'http_1_1_close_workarounds', 'idleTimeout': 'idle_timeout', 'insertXforwardedFor': 'insert_xforwarded_for', 'maxHeaderSize': 'maximum_header_size', 'maxRequests': 'maximum_requests', 'mssOverride': 'maximum_segment_size_override', 'receiveWindowSize': 'receive_window_size', 'resetOnTimeout': 'reset_on_timeout', 'serverCloseTimeout': 'server_close_timeout', 'serverSack': 'server_sack', 'serverTimestamp': 'server_timestamp', 'uncleanShutdown': 'unclean_shutdown'}
    returnables = ['full_path', 'name', 'client_close_timeout', 'oneconnect_idle_timeout_override', 'oneconnect_maximum_reuse', 'oneconnect_maximum_pool_size', 'oneconnect_minimum_pool_size', 'oneconnect_replenish', 'oneconnect_ramp_up_increment', 'parent', 'description', 'force_http_1_0_response', 'request_header_insert', 'http_1_1_close_workarounds', 'idle_timeout', 'insert_xforwarded_for', 'maximum_header_size', 'maximum_requests', 'maximum_segment_size_override', 'receive_window_size', 'reset_on_timeout', 'server_close_timeout', 'server_sack', 'server_timestamp', 'unclean_shutdown']

    @property
    def request_header_insert(self):
        if self._values['request_header_insert'] in [None, 'none']:
            return None
        return self._values['request_header_insert']

    @property
    def server_timestamp(self):
        return flatten_boolean(self._values['server_timestamp'])

    @property
    def server_sack(self):
        return flatten_boolean(self._values['server_sack'])

    @property
    def reset_on_timeout(self):
        return flatten_boolean(self._values['reset_on_timeout'])

    @property
    def insert_xforwarded_for(self):
        return flatten_boolean(self._values['insert_xforwarded_for'])

    @property
    def http_1_1_close_workarounds(self):
        return flatten_boolean(self._values['http_1_1_close_workarounds'])

    @property
    def force_http_1_0_response(self):
        return flatten_boolean(self._values['force_http_1_0_response'])

    @property
    def oneconnect_replenish(self):
        return flatten_boolean(self._values['oneconnect_replenish'])

    @property
    def idle_timeout(self):
        if self._values['idle_timeout'] is None:
            return None
        elif self._values['idle_timeout'] == 'immediate':
            return 0
        elif self._values['idle_timeout'] == 'indefinite':
            return 4294967295
        return int(self._values['idle_timeout'])