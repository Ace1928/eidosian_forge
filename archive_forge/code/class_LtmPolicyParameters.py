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
class LtmPolicyParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'rulesReference': 'rules'}
    returnables = ['full_path', 'name', 'status', 'description', 'strategy', 'rules', 'requires', 'controls']

    def _handle_conditions(self, conditions):
        result = []
        if conditions is None or 'items' not in conditions:
            return result
        for condition in conditions['items']:
            tmp = dict()
            tmp['case_insensitive'] = flatten_boolean(condition.pop('caseInsensitive', None))
            tmp['case_sensitive'] = flatten_boolean(condition.pop('caseSensitive', None))
            tmp['contains_string'] = flatten_boolean(condition.pop('contains', None))
            tmp['external'] = flatten_boolean(condition.pop('external', None))
            tmp['http_basic_auth'] = flatten_boolean(condition.pop('httpBasicAuth', None))
            tmp['http_host'] = flatten_boolean(condition.pop('httpHost', None))
            tmp['datagroup'] = condition.pop('datagroup', None)
            tmp['tcp'] = flatten_boolean(condition.pop('tcp', None))
            tmp['remote'] = flatten_boolean(condition.pop('remote', None))
            tmp['matches'] = flatten_boolean(condition.pop('matches', None))
            tmp['address'] = flatten_boolean(condition.pop('address', None))
            tmp['present'] = flatten_boolean(condition.pop('present', None))
            tmp['proxy_connect'] = flatten_boolean(condition.pop('proxyConnect', None))
            tmp['proxy_request'] = flatten_boolean(condition.pop('proxyRequest', None))
            tmp['host'] = flatten_boolean(condition.pop('host', None))
            tmp['http_uri'] = flatten_boolean(condition.pop('httpUri', None))
            tmp['request'] = flatten_boolean(condition.pop('request', None))
            tmp['username'] = flatten_boolean(condition.pop('username', None))
            tmp['external'] = flatten_boolean(condition.pop('external', None))
            tmp['values'] = condition.pop('values', None)
            tmp['all'] = flatten_boolean(condition.pop('all', None))
            result.append(self._filter_params(tmp))
        return result

    def _handle_actions(self, actions):
        result = []
        if actions is None or 'items' not in actions:
            return result
        exclude_keys = ['poolReference', 'name']
        for action in actions['items']:
            tmp = dict(((k, v) for k, v in iteritems(action) if v != 0 and k not in exclude_keys))
            self._remove_internal_keywords(tmp)
            result.append(self._filter_params(tmp))
        return result

    @property
    def rules(self):
        result = []
        if self._values['rules'] is None or 'items' not in self._values['rules']:
            return result
        for item in self._values['rules']['items']:
            self._remove_internal_keywords(item)
            item['conditions'] = self._handle_conditions(item.pop('conditionsReference', None))
            item['actions'] = self._handle_actions(item.pop('actionsReference', None))
            result.append(item)
        return result