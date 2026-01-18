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
class GtmServersParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'exposeRouteDomains': 'expose_route_domains', 'iqAllowPath': 'iq_allow_path', 'iqAllowServiceCheck': 'iq_allow_service_check', 'iqAllowSnmp': 'iq_allow_snmp', 'limitCpuUsage': 'limit_cpu_usage', 'limitCpuUsageStatus': 'limit_cpu_usage_status', 'limitMaxBps': 'limit_max_bps', 'limitMaxBpsStatus': 'limit_max_bps_status', 'limitMaxConnections': 'limit_max_connections', 'limitMaxConnectionsStatus': 'limit_max_connections_status', 'limitMaxPps': 'limit_max_pps', 'limitMaxPpsStatus': 'limit_max_pps_status', 'limitMemAvail': 'limit_mem_available', 'limitMemAvailStatus': 'limit_mem_available_status', 'linkDiscovery': 'link_discovery', 'proberFallback': 'prober_fallback', 'proberPreference': 'prober_preference', 'virtualServerDiscovery': 'virtual_server_discovery', 'devicesReference': 'devices', 'virtualServersReference': 'virtual_servers', 'monitor': 'monitors'}
    returnables = ['datacenter', 'enabled', 'disabled', 'expose_route_domains', 'iq_allow_path', 'full_path', 'iq_allow_service_check', 'iq_allow_snmp', 'limit_cpu_usage', 'limit_cpu_usage_status', 'limit_max_bps', 'limit_max_bps_status', 'limit_max_connections', 'limit_max_connections_status', 'limit_max_pps', 'limit_max_pps_status', 'limit_mem_available', 'limit_mem_available_status', 'link_discovery', 'monitors', 'monitor_type', 'name', 'product', 'prober_fallback', 'prober_preference', 'virtual_server_discovery', 'addresses', 'devices', 'virtual_servers']

    def _remove_internal_keywords(self, resource, stats=False):
        if stats:
            resource.pop('kind', None)
            resource.pop('generation', None)
            resource.pop('isSubcollection', None)
            resource.pop('fullPath', None)
        else:
            resource.pop('kind', None)
            resource.pop('generation', None)
            resource.pop('selfLink', None)
            resource.pop('isSubcollection', None)
            resource.pop('fullPath', None)

    def _read_virtual_stats_from_device(self, url):
        uri = 'https://{0}:{1}{2}/stats'.format(self.client.provider['server'], self.client.provider['server_port'], url)
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
            raise F5ModuleError(resp.content)
        result = parseStats(response)
        try:
            return result['stats']
        except KeyError:
            return {}

    def _process_vs_stats(self, link):
        result = dict()
        item = self._read_virtual_stats_from_device(urlparse(link).path)
        if not item:
            return result
        result['status'] = item['status']['availabilityState']
        result['status_reason'] = item['status']['statusReason']
        result['state'] = item['status']['enabledState']
        result['bits_per_sec_in'] = item['metrics']['bitsPerSecIn']
        result['bits_per_sec_in'] = item['metrics']['bitsPerSecOut']
        result['pkts_per_sec_in'] = item['metrics']['pktsPerSecIn']
        result['pkts_per_sec_out'] = item['metrics']['pktsPerSecOut']
        result['connections'] = item['metrics']['connections']
        result['picks'] = item['picks']
        result['virtual_server_score'] = item['metrics']['vsScore']
        result['uptime'] = item['uptime']
        return result

    @property
    def monitors(self):
        if self._values['monitors'] is None:
            return []
        try:
            result = re.findall('/\\w+/[^\\s}]+', self._values['monitors'])
            return result
        except Exception:
            return [self._values['monitors']]

    @property
    def monitor_type(self):
        if self._values['monitors'] is None:
            return None
        pattern = 'min\\s+\\d+\\s+of'
        matches = re.search(pattern, self._values['monitors'])
        if matches:
            return 'm_of_n'
        else:
            return 'and_list'

    @property
    def limit_mem_available_status(self):
        return flatten_boolean(self._values['limit_mem_available_status'])

    @property
    def limit_max_pps_status(self):
        return flatten_boolean(self._values['limit_max_pps_status'])

    @property
    def limit_max_connections_status(self):
        return flatten_boolean(self._values['limit_max_connections_status'])

    @property
    def limit_max_bps_status(self):
        return flatten_boolean(self._values['limit_max_bps_status'])

    @property
    def limit_cpu_usage_status(self):
        return flatten_boolean(self._values['limit_cpu_usage_status'])

    @property
    def iq_allow_service_check(self):
        return flatten_boolean(self._values['iq_allow_service_check'])

    @property
    def iq_allow_snmp(self):
        return flatten_boolean(self._values['iq_allow_snmp'])

    @property
    def expose_route_domains(self):
        return flatten_boolean(self._values['expose_route_domains'])

    @property
    def iq_allow_path(self):
        return flatten_boolean(self._values['iq_allow_path'])

    @property
    def product(self):
        if self._values['product'] is None:
            return None
        if self._values['product'] in ['single-bigip', 'redundant-bigip']:
            return 'bigip'
        return self._values['product']

    @property
    def devices(self):
        result = []
        if self._values['devices'] is None or 'items' not in self._values['devices']:
            return result
        for item in self._values['devices']['items']:
            self._remove_internal_keywords(item)
            if 'fullPath' in item:
                item['full_path'] = item.pop('fullPath')
            result.append(item)
        return result

    @property
    def virtual_servers(self):
        result = []
        if self._values['virtual_servers'] is None or 'items' not in self._values['virtual_servers']:
            return result
        for item in self._values['virtual_servers']['items']:
            self._remove_internal_keywords(item, stats=True)
            stats = self._process_vs_stats(item['selfLink'])
            self._remove_internal_keywords(item)
            item['stats'] = stats
            if 'disabled' in item:
                if item['disabled'] in BOOLEANS_TRUE:
                    item['disabled'] = flatten_boolean(item['disabled'])
                    item['enabled'] = flatten_boolean(not item['disabled'])
            if 'enabled' in item:
                if item['enabled'] in BOOLEANS_TRUE:
                    item['enabled'] = flatten_boolean(item['enabled'])
                    item['disabled'] = flatten_boolean(not item['enabled'])
            if 'fullPath' in item:
                item['full_path'] = item.pop('fullPath')
            if 'limitMaxBps' in item:
                item['limit_max_bps'] = int(item.pop('limitMaxBps'))
            if 'limitMaxBpsStatus' in item:
                item['limit_max_bps_status'] = item.pop('limitMaxBpsStatus')
            if 'limitMaxConnections' in item:
                item['limit_max_connections'] = int(item.pop('limitMaxConnections'))
            if 'limitMaxConnectionsStatus' in item:
                item['limit_max_connections_status'] = item.pop('limitMaxConnectionsStatus')
            if 'limitMaxPps' in item:
                item['limit_max_pps'] = int(item.pop('limitMaxPps'))
            if 'limitMaxPpsStatus' in item:
                item['limit_max_pps_status'] = item.pop('limitMaxPpsStatus')
            if 'translationAddress' in item:
                item['translation_address'] = item.pop('translationAddress')
            if 'translationPort' in item:
                item['translation_port'] = int(item.pop('translationPort'))
            result.append(item)
        return result

    @property
    def limit_cpu_usage(self):
        if self._values['limit_cpu_usage'] is None:
            return None
        return int(self._values['limit_cpu_usage'])

    @property
    def limit_max_bps(self):
        if self._values['limit_max_bps'] is None:
            return None
        return int(self._values['limit_max_bps'])

    @property
    def limit_max_connections(self):
        if self._values['limit_max_connections'] is None:
            return None
        return int(self._values['limit_max_connections'])

    @property
    def limit_max_pps(self):
        if self._values['limit_max_pps'] is None:
            return None
        return int(self._values['limit_max_pps'])

    @property
    def limit_mem_available(self):
        if self._values['limit_mem_available'] is None:
            return None
        return int(self._values['limit_mem_available'])