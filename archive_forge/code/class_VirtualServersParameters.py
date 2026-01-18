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
class VirtualServersParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'autoLasthop': 'auto_lasthop', 'bwcPolicy': 'bw_controller_policy', 'cmpEnabled': 'cmp_enabled', 'connectionLimit': 'connection_limit', 'fallbackPersistence': 'fallback_persistence_profile', 'persist': 'persistence_profile', 'translatePort': 'translate_port', 'translateAddress': 'translate_address', 'lastHopPool': 'last_hop_pool', 'nat64': 'nat64_enabled', 'sourcePort': 'source_port_behavior', 'ipIntelligencePolicy': 'ip_intelligence_policy', 'ipProtocol': 'protocol', 'pool': 'default_pool', 'rateLimitMode': 'rate_limit_mode', 'rateLimitSrcMask': 'rate_limit_source_mask', 'rateLimitDstMask': 'rate_limit_destination_mask', 'rateLimit': 'rate_limit', 'sourceAddressTranslation': 'snat_type', 'gtmScore': 'gtm_score', 'rateClass': 'rate_class', 'source': 'source_address', 'auth': 'authentication_profile', 'mirror': 'connection_mirror_enabled', 'rules': 'irules', 'securityLogProfiles': 'security_log_profiles', 'profilesReference': 'profiles', 'policiesReference': 'policies'}
    returnables = ['full_path', 'name', 'auto_lasthop', 'bw_controller_policy', 'cmp_enabled', 'connection_limit', 'description', 'enabled', 'fallback_persistence_profile', 'persistence_profile', 'translate_port', 'translate_address', 'vlans', 'destination', 'last_hop_pool', 'nat64_enabled', 'source_port_behavior', 'ip_intelligence_policy', 'protocol', 'default_pool', 'rate_limit_mode', 'rate_limit_source_mask', 'rate_limit', 'snat_type', 'snat_pool', 'gtm_score', 'rate_class', 'rate_limit_destination_mask', 'source_address', 'authentication_profile', 'connection_mirror_enabled', 'irules', 'security_log_profiles', 'type', 'policies', 'profiles', 'destination_address', 'destination_port', 'availability_status', 'status_reason', 'total_requests', 'client_side_bits_in', 'client_side_bits_out', 'client_side_current_connections', 'client_side_evicted_connections', 'client_side_max_connections', 'client_side_pkts_in', 'client_side_pkts_out', 'client_side_slow_killed', 'client_side_total_connections', 'cmp_mode', 'ephemeral_bits_in', 'ephemeral_bits_out', 'ephemeral_current_connections', 'ephemeral_evicted_connections', 'ephemeral_max_connections', 'ephemeral_pkts_in', 'ephemeral_pkts_out', 'ephemeral_slow_killed', 'ephemeral_total_connections', 'total_software_accepted_syn_cookies', 'total_hardware_accepted_syn_cookies', 'total_hardware_syn_cookies', 'hardware_syn_cookie_instances', 'total_software_rejected_syn_cookies', 'software_syn_cookie_instances', 'current_syn_cache', 'syn_cache_overflow', 'total_software_syn_cookies', 'syn_cookies_status', 'max_conn_duration', 'mean_conn_duration', 'min_conn_duration', 'cpu_usage_ratio_last_5_min', 'cpu_usage_ratio_last_5_sec', 'cpu_usage_ratio_last_1_min']

    @property
    def max_conn_duration(self):
        return self._values['stats']['csMaxConnDur']

    @property
    def mean_conn_duration(self):
        return self._values['stats']['csMeanConnDur']

    @property
    def min_conn_duration(self):
        return self._values['stats']['csMinConnDur']

    @property
    def cpu_usage_ratio_last_5_min(self):
        return self._values['stats']['fiveMinAvgUsageRatio']

    @property
    def cpu_usage_ratio_last_5_sec(self):
        return self._values['stats']['fiveSecAvgUsageRatio']

    @property
    def cpu_usage_ratio_last_1_min(self):
        return self._values['stats']['oneMinAvgUsageRatio']

    @property
    def cmp_mode(self):
        return self._values['stats']['cmpEnableMode']

    @property
    def availability_status(self):
        return self._values['stats']['status']['availabilityState']

    @property
    def status_reason(self):
        return self._values['stats']['status']['statusReason']

    @property
    def total_requests(self):
        return self._values['stats']['totRequests']

    @property
    def ephemeral_bits_in(self):
        return self._values['stats']['ephemeral']['bitsIn']

    @property
    def ephemeral_bits_out(self):
        return self._values['stats']['ephemeral']['bitsOut']

    @property
    def ephemeral_current_connections(self):
        return self._values['stats']['ephemeral']['curConns']

    @property
    def ephemeral_evicted_connections(self):
        return self._values['stats']['ephemeral']['evictedConns']

    @property
    def ephemeral_max_connections(self):
        return self._values['stats']['ephemeral']['maxConns']

    @property
    def ephemeral_pkts_in(self):
        return self._values['stats']['ephemeral']['pktsIn']

    @property
    def ephemeral_pkts_out(self):
        return self._values['stats']['ephemeral']['pktsOut']

    @property
    def ephemeral_slow_killed(self):
        return self._values['stats']['ephemeral']['slowKilled']

    @property
    def ephemeral_total_connections(self):
        return self._values['stats']['ephemeral']['totConns']

    @property
    def client_side_bits_in(self):
        return self._values['stats']['clientside']['bitsIn']

    @property
    def client_side_bits_out(self):
        return self._values['stats']['clientside']['bitsOut']

    @property
    def client_side_current_connections(self):
        return self._values['stats']['clientside']['curConns']

    @property
    def client_side_evicted_connections(self):
        return self._values['stats']['clientside']['evictedConns']

    @property
    def client_side_max_connections(self):
        return self._values['stats']['clientside']['maxConns']

    @property
    def client_side_pkts_in(self):
        return self._values['stats']['clientside']['pktsIn']

    @property
    def client_side_pkts_out(self):
        return self._values['stats']['clientside']['pktsOut']

    @property
    def client_side_slow_killed(self):
        return self._values['stats']['clientside']['slowKilled']

    @property
    def client_side_total_connections(self):
        return self._values['stats']['clientside']['totConns']

    @property
    def total_software_accepted_syn_cookies(self):
        return self._values['stats']['syncookie']['accepts']

    @property
    def total_hardware_accepted_syn_cookies(self):
        return self._values['stats']['syncookie']['hwAccepts']

    @property
    def total_hardware_syn_cookies(self):
        return self._values['stats']['syncookie']['hwSyncookies']

    @property
    def hardware_syn_cookie_instances(self):
        return self._values['stats']['syncookie']['hwsyncookieInstance']

    @property
    def total_software_rejected_syn_cookies(self):
        return self._values['stats']['syncookie']['rejects']

    @property
    def software_syn_cookie_instances(self):
        return self._values['stats']['syncookie']['swsyncookieInstance']

    @property
    def current_syn_cache(self):
        return self._values['stats']['syncookie']['syncacheCurr']

    @property
    def syn_cache_overflow(self):
        return self._values['stats']['syncookie']['syncacheOver']

    @property
    def total_software_syn_cookies(self):
        return self._values['stats']['syncookie']['syncookies']

    @property
    def syn_cookies_status(self):
        return self._values['stats']['syncookieStatus']

    @property
    def destination_address(self):
        if self._values['destination'] is None:
            return None
        tup = self.destination_tuple
        return tup.ip

    @property
    def destination_port(self):
        if self._values['destination'] is None:
            return None
        tup = self.destination_tuple
        return tup.port

    @property
    def type(self):
        """Attempt to determine the current server type

        This check is very unscientific. It turns out that this information is not
        exactly available anywhere on a BIG-IP. Instead, we rely on a semi-reliable
        means for determining what the type of the virtual server is. Hopefully it
        always works.

        There are a handful of attributes that can be used to determine a specific
        type. There are some types though that can only be determined by looking at
        the profiles that are assigned to them. We follow that method for those
        complicated types; message-routing, fasthttp, and fastl4.

        Because type determination is an expensive operation, we cache the result
        from the operation.

        Returns:
            string: The server type.
        """
        if self._values['l2Forward'] is True:
            result = 'forwarding-l2'
        elif self._values['ipForward'] is True:
            result = 'forwarding-ip'
        elif self._values['stateless'] is True:
            result = 'stateless'
        elif self._values['reject'] is True:
            result = 'reject'
        elif self._values['dhcpRelay'] is True:
            result = 'dhcp'
        elif self._values['internal'] is True:
            result = 'internal'
        elif self.has_fasthttp_profiles:
            result = 'performance-http'
        elif self.has_fastl4_profiles:
            result = 'performance-l4'
        elif self.has_message_routing_profiles:
            result = 'message-routing'
        else:
            result = 'standard'
        return result

    @property
    def profiles(self):
        """Returns a list of profiles from the API

        The profiles are formatted so that they are usable in this module and
        are able to be compared by the Difference engine.

        Returns:
             list (:obj:`list` of :obj:`dict`): List of profiles.

             Each dictionary in the list contains the following three (3) keys.

             * name
             * context
             * fullPath

        Raises:
            F5ModuleError: If the specified context is a value other that
                ``all``, ``server-side``, or ``client-side``.
        """
        if 'items' not in self._values['profiles']:
            return None
        result = []
        for item in self._values['profiles']['items']:
            context = item['context']
            if context == 'serverside':
                context = 'server-side'
            elif context == 'clientside':
                context = 'client-side'
            name = item['name']
            if context in ['all', 'server-side', 'client-side']:
                result.append(dict(name=name, context=context, full_path=item['fullPath']))
            else:
                raise F5ModuleError("Unknown profile context found: '{0}'".format(context))
        return result

    @property
    def has_message_routing_profiles(self):
        if self.profiles is None:
            return None
        current = self._read_current_message_routing_profiles_from_device()
        result = [x['name'] for x in self.profiles if x['name'] in current]
        if len(result) > 0:
            return True
        return False

    @property
    def has_fastl4_profiles(self):
        if self.profiles is None:
            return None
        current = self._read_current_fastl4_profiles_from_device()
        result = [x['name'] for x in self.profiles if x['name'] in current]
        if len(result) > 0:
            return True
        return False

    @property
    def has_fasthttp_profiles(self):
        """Check if ``fasthttp`` profile is in API profiles

        This method is used to determine the server type when doing comparisons
        in the Difference class.

        Returns:
             bool: True if server has ``fasthttp`` profiles. False otherwise.
        """
        if self.profiles is None:
            return None
        current = self._read_current_fasthttp_profiles_from_device()
        result = [x['name'] for x in self.profiles if x['name'] in current]
        if len(result) > 0:
            return True
        return False

    def _read_current_message_routing_profiles_from_device(self):
        result = []
        result += self._read_diameter_profiles_from_device()
        result += self._read_sip_profiles_from_device()
        return result

    def _read_diameter_profiles_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/ltm/profile/diameter/'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
            raise F5ModuleError(resp.content)
        result = [x['name'] for x in response['items']]
        return result

    def _read_sip_profiles_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/ltm/profile/sip/'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
            raise F5ModuleError(resp.content)
        result = [x['name'] for x in response['items']]
        return result

    def _read_current_fastl4_profiles_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/ltm/profile/fastl4/'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
            raise F5ModuleError(resp.content)
        result = [x['name'] for x in response['items']]
        return result

    def _read_current_fasthttp_profiles_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/ltm/profile/fasthttp/'.format(self.client.provider['server'], self.client.provider['server_port'])
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
            raise F5ModuleError(resp.content)
        result = [x['name'] for x in response['items']]
        return result

    @property
    def security_log_profiles(self):
        if self._values['security_log_profiles'] is None:
            return None
        result = list(set([x.strip('"') for x in self._values['security_log_profiles']]))
        result.sort()
        return result

    @property
    def snat_type(self):
        if self._values['snat_type'] is None:
            return None
        if 'type' in self._values['snat_type']:
            if self._values['snat_type']['type'] == 'automap':
                return 'automap'
            elif self._values['snat_type']['type'] == 'none':
                return 'none'
            elif self._values['snat_type']['type'] == 'snat':
                return 'snat'

    @property
    def snat_pool(self):
        if self._values['snat_type'] is None:
            return None
        if 'type' in self._values['snat_type']:
            if self._values['snat_type']['type'] == 'automap':
                return 'none'
            elif self._values['snat_type']['type'] == 'none':
                return 'none'
            elif self._values['snat_type']['type'] == 'snat':
                return self._values['snat_type']['pool']

    @property
    def connection_mirror_enabled(self):
        if self._values['connection_mirror_enabled'] is None:
            return None
        elif self._values['connection_mirror_enabled'] == 'enabled':
            return 'yes'
        return 'no'

    @property
    def rate_limit(self):
        if self._values['rate_limit'] is None:
            return None
        elif self._values['rate_limit'] == 'disabled':
            return -1
        return int(self._values['rate_limit'])

    @property
    def nat64_enabled(self):
        if self._values['nat64_enabled'] is None:
            return None
        elif self._values['nat64_enabled'] == 'enabled':
            return 'yes'
        return 'no'

    @property
    def enabled(self):
        if self._values['enabled'] is None:
            return 'no'
        elif self._values['enabled'] is True:
            return 'yes'
        return 'no'

    @property
    def translate_port(self):
        if self._values['translate_port'] is None:
            return None
        elif self._values['translate_port'] == 'enabled':
            return 'yes'
        return 'no'

    @property
    def translate_address(self):
        if self._values['translate_address'] is None:
            return None
        elif self._values['translate_address'] == 'enabled':
            return 'yes'
        return 'no'

    @property
    def persistence_profile(self):
        """Return persistence profile in a consumable form

        I don't know why the persistence profile is stored this way, but below is the
        general format of it.

            "persist": [
                {
                    "name": "msrdp",
                    "partition": "Common",
                    "tmDefault": "yes",
                    "nameReference": {
                        "link": "https://localhost/mgmt/tm/ltm/persistence/msrdp/~Common~msrdp?ver=13.1.0.4"
                    }
                }
            ],

        As you can see, this is quite different from something like the fallback
        persistence profile which is just simply

            /Common/fallback1

        This method makes the persistence profile look like the fallback profile.

        Returns:
             string: The persistence profile configured on the virtual.
        """
        if self._values['persistence_profile'] is None:
            return None
        profile = self._values['persistence_profile'][0]
        result = fq_name(profile['partition'], profile['name'])
        return result

    @property
    def destination_tuple(self):
        Destination = namedtuple('Destination', ['ip', 'port', 'route_domain', 'mask'])
        if self._values['destination'] is None:
            result = Destination(ip=None, port=None, route_domain=None, mask=None)
            return result
        destination = re.sub('^/[a-zA-Z0-9_.-]+/([a-zA-Z0-9_.-]+\\/)?', '', self._values['destination'])
        pattern = '(?P<ip>[^%]+)%(?P<route_domain>[0-9]+)[:.](?P<port>[0-9]+|any)'
        matches = re.search(pattern, destination)
        if matches:
            try:
                port = int(matches.group('port'))
            except ValueError:
                port = matches.group('port')
                if port == 'any':
                    port = 0
            result = Destination(ip=matches.group('ip'), port=port, route_domain=int(matches.group('route_domain')), mask=self.mask)
            return result
        pattern = '(?P<ip>[^%]+)%(?P<route_domain>[0-9]+)'
        matches = re.search(pattern, destination)
        if matches:
            result = Destination(ip=matches.group('ip'), port=None, route_domain=int(matches.group('route_domain')), mask=self.mask)
            return result
        pattern = '^(?P<ip>(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])):(?P<port>[0-9]+)'
        matches = re.search(pattern, destination)
        if matches:
            result = Destination(ip=matches.group('ip'), port=int(matches.group('port')), route_domain=None, mask=self.mask)
            return result
        pattern = '^([0-9a-f]{0,4}:){2,7}(:|[0-9a-f]{1,4})$'
        matches = re.search(pattern, destination)
        if matches:
            result = Destination(ip=destination, port=None, route_domain=None, mask=self.mask)
            return result
        pattern = '(?P<ip>([0-9a-f]{0,4}:){2,7}(:|[0-9a-f]{1,4}).(?P<port>[0-9]+|any))'
        matches = re.search(pattern, destination)
        if matches:
            ip = matches.group('ip').split('.')[0]
            try:
                port = int(matches.group('port'))
            except ValueError:
                port = matches.group('port')
                if port == 'any':
                    port = 0
            result = Destination(ip=ip, port=port, route_domain=None, mask=self.mask)
            return result
        pattern = '(?P<name>^[a-zA-Z0-9_.-]+):(?P<port>[0-9]+)'
        matches = re.search(pattern, destination)
        if matches:
            result = Destination(ip=matches.group('name'), port=int(matches.group('port')), route_domain=None, mask=self.mask)
            return result
        pattern = '(?P<name>^[a-zA-Z0-9_.-]+)'
        matches = re.search(pattern, destination)
        if matches:
            result = Destination(ip=matches.group('name'), port=None, route_domain=None, mask=self.mask)
            return result
        pattern = '(?P<ip>[^.]+).(?P<port>[0-9]+|any)'
        matches = re.search(pattern, destination)
        if matches:
            result = Destination(ip=matches.group('ip'), port=matches.group('port'), route_domain=None, mask=self.mask)
            return result
        result = Destination(ip=None, port=None, route_domain=None, mask=None)
        return result

    @property
    def policies(self):
        if 'items' not in self._values['policies']:
            return None
        results = []
        for item in self._values['policies']['items']:
            results.append(item['fullPath'])
        return results