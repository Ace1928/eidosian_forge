from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
from datetime import datetime
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import (
from ipaddress import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import (
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip_interface
from ..module_utils.teem import send_teem
class RecordsEncoder(object):

    def __init__(self, record_type=None, separator=None):
        self._record_type = record_type
        self._separator = separator
        self._network_pattern = re.compile('^network\\s+(?P<addr>[^ ]+)\\s+prefixlen\\s+(?P<prefix>\\d+)\\s+.*')
        self._rd_net_prefix_ptrn = re.compile('^network\\s+(?P<addr>[^%]+)%(?P<rd>[0-9]+)\\s+prefixlen\\s+(?P<prefix>\\d+)\\s+.*')
        self._host_pattern = re.compile('^host\\s+(?P<addr>[^%]+)\\s+.*')
        self._rd_host_ptrn = re.compile('^host\\s+(?P<addr>[^%]+)%(?P<rd>[0-9]+)\\s+.*')
        self._ipv4_cidr_ptrn = re.compile('^(?P<addr>((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?))/(?P<cidr>(3[0-2]|2[0-9]|1[0-9]|[0-9]))')
        self._ipv4_cidr_ptrn_rd = re.compile('^(?P<addr>((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?))%(?P<rd>[0-9]+)/(?P<cidr>(3[0-2]|2[0-9]|1[0-9]|[0-9]))')
        self._ipv6_cidr_ptrn = re.compile('^(?P<addr>^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])))/(?P<cidr>((1(1[0-9]|2[0-8]))|([0-9][0-9])|([0-9])))')
        self._ipv6_cidr_ptrn_rd = re.compile('^(?P<addr>^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])))%(?P<rd>[0-9]+)/(?P<cidr>((1(1[0-9]|2[0-8]))|([0-9][0-9])|([0-9])))')

    def encode(self, record):
        if isinstance(record, dict):
            return self.encode_dict(record)
        else:
            return self.encode_string(record)

    def encode_dict(self, record):
        if self._record_type == 'ip':
            return self.encode_address_from_dict(record)
        elif self._record_type == 'integer':
            return self.encode_integer_from_dict(record)
        else:
            return self.encode_string_from_dict(record)

    def encode_rd_address(self, record, match, ipv6=False):
        if is_valid_ip_interface(match.group('addr')):
            key = ip_interface(u'{0}/{1}'.format(match.group('addr'), match.group('cidr')))
        else:
            raise F5ModuleError("When specifying an 'address' type, the value to the left of the separator must be an IP.")
        if key and 'value' in record:
            if ipv6 and key.network.prefixlen == 128:
                return self.encode_host(str(key.ip) + '%' + match.group('rd'), record['value'])
            elif not ipv6 and key.network.prefixlen == 32:
                return self.encode_host(str(key.ip) + '%' + match.group('rd'), record['value'])
            return self.encode_network(str(key.network.network_address) + '%' + match.group('rd'), key.network.prefixlen, record['value'])
        elif key:
            if ipv6 and key.network.prefixlen == 128:
                return self.encode_host(str(key.ip) + '%' + match.group('rd'), str(key.ip) + '%' + match.group('rd'))
            elif not ipv6 and key.network.prefixlen == 32:
                return self.encode_host(str(key.ip) + '%' + match.group('rd'), str(key.ip) + '%' + match.group('rd'))
            return self.encode_network(str(key.network.network_address) + '%' + match.group('rd'), key.network.prefixlen, str(key.network.network_address) + '%' + match.group('rd'))

    def encode_address_from_dict(self, record):
        rd_match = re.match(self._ipv4_cidr_ptrn_rd, record['key'])
        if rd_match:
            return self.encode_rd_address(record, rd_match)
        rd_match = re.match(self._ipv6_cidr_ptrn_rd, record['key'])
        if rd_match:
            return self.encode_rd_address(record, rd_match, ipv6=True)
        if is_valid_ip_interface(record['key']):
            key = ip_interface(u'{0}'.format(str(record['key'])))
        else:
            raise F5ModuleError("When specifying an 'address' type, the value to the left of the separator must be an IP.")
        ipv4_match = re.match(self._ipv4_cidr_ptrn, record['key'])
        ipv6_match = re.match(self._ipv6_cidr_ptrn, record['key'])
        if key and 'value' in record:
            if ipv6_match and key.network.prefixlen == 128 or (ipv4_match and key.network.prefixlen == 32):
                return self.encode_host(str(key.ip), record['value'])
            else:
                return self.encode_network(str(key.network.network_address), key.network.prefixlen, record['value'])
        elif key:
            if ipv6_match and key.network.prefixlen == 128 or (ipv4_match and key.network.prefixlen == 32):
                return self.encode_host(str(key.ip), str(key.ip))
            else:
                return self.encode_network(str(key.network.network_address), key.network.prefixlen, str(key.network.network_address))

    def encode_integer_from_dict(self, record):
        try:
            int(record['key'])
        except ValueError:
            raise F5ModuleError("When specifying an 'integer' type, the value to the left of the separator must be a number.")
        if 'key' in record and 'value' in record:
            return '{0} {1} {2}'.format(record['key'], self._separator, record['value'])
        elif 'key' in record:
            return str(record['key'])

    def encode_string_from_dict(self, record):
        if 'key' in record and 'value' in record:
            return '{0} {1} {2}'.format(record['key'], self._separator, record['value'])
        elif 'key' in record:
            return '{0} {1} ""'.format(record['key'], self._separator)

    def encode_string(self, record):
        record = record.strip().strip(',')
        if self._record_type == 'ip':
            return self.encode_address_from_string(record)
        elif self._record_type == 'integer':
            return self.encode_integer_from_string(record)
        else:
            return self.encode_string_from_string(record)

    def encode_address_from_string(self, record):
        if self._network_pattern.match(record):
            return record
        elif self._host_pattern.match(record):
            return record
        elif self._rd_net_prefix_ptrn.match(record) or self._rd_host_ptrn.match(record):
            return record
        elif self._ipv4_cidr_ptrn_rd.match(record) or self._ipv6_cidr_ptrn_rd.match(record):
            parts = [r.strip() for r in record.split(self._separator)]
            if parts[0] == '':
                return
            pattern = re.compile('(?P<addr>[^%]+)%(?P<rd>[0-9]+)/(?P<prefix>[0-9]+)')
            match = pattern.match(parts[0])
            addr = u'{0}/{1}'.format(match.group('addr'), match.group('prefix'))
            if not is_valid_ip_interface(addr):
                raise F5ModuleError("When specifying an 'address' type, the value to the left of the separator must be an IP.")
            key = ip_interface(addr)
            ipv4_match = re.match(self._ipv4_cidr_ptrn, addr)
            ipv6_match = re.match(self._ipv6_cidr_ptrn, addr)
            if len(parts) == 2:
                if ipv4_match and key.network.prefixlen == 32 or (ipv6_match and key.network.prefixlen == 128):
                    return self.encode_host(str(key.ip) + '%' + str(match.group('rd')), parts[1])
                else:
                    return self.encode_network(str(key.network.network_address) + '%' + str(match.group('rd')), key.network.prefixlen, parts[1])
            elif len(parts) == 1 and parts[0] != '':
                if ipv4_match and key.network.prefixlen == 32 or (ipv6_match and key.network.prefixlen == 128):
                    return self.encode_host(str(key.ip) + '%' + str(match.group('rd')), str(key.ip) + '%' + str(match.group('rd')))
                return self.encode_network(str(key.network.network_address) + '%' + str(match.group('rd')), key.network.prefixlen, str(key.network.network_address) + '%' + str(match.group('rd')))
        else:
            parts = [r.strip() for r in record.split(self._separator)]
            if parts[0] == '':
                return
            if len(re.split(' ', parts[0])) == 1:
                if not is_valid_ip_interface(parts[0]):
                    raise F5ModuleError("When specifying an 'address' type, the value to the left of the separator must be an IP.")
                key = ip_interface(u'{0}'.format(str(parts[0])))
                ipv4_match = re.match(self._ipv4_cidr_ptrn, str(parts[0]))
                ipv6_match = re.match(self._ipv6_cidr_ptrn, str(parts[0]))
                if len(parts) == 2:
                    if ipv4_match and key.network.prefixlen == 32 or (ipv6_match and key.network.prefixlen == 128):
                        return self.encode_host(str(key.ip), parts[1])
                    else:
                        return self.encode_network(str(key.network.network_address), key.network.prefixlen, parts[1])
                elif len(parts) == 1 and parts[0] != '':
                    if ipv4_match and key.network.prefixlen == 32 or (ipv6_match and key.network.prefixlen == 128):
                        return self.encode_host(str(key.ip), str(key.ip))
                    return self.encode_network(str(key.network.network_address), key.network.prefixlen, str(key.network.network_address))
            else:
                return str(parts[0])

    def encode_host(self, key, value):
        return 'host {0} {1} {2}'.format(str(key), self._separator, str(value))

    def encode_network(self, key, prefixlen, value):
        return 'network {0} prefixlen {1} {2} {3}'.format(str(key), str(prefixlen), self._separator, str(value))

    def encode_integer_from_string(self, record):
        parts = record.split(self._separator)
        if len(parts) == 1 and parts[0] == '':
            return None
        try:
            int(parts[0])
        except ValueError:
            raise F5ModuleError("When specifying an 'integer' type, the value to the left of the separator must be a number.")
        if len(parts) == 2:
            return '{0} {1} {2}'.format(parts[0], self._separator, parts[1])
        elif len(parts) == 1:
            return str(parts[0])

    def encode_string_from_string(self, record):
        parts = record.split(self._separator)
        if len(parts) == 2:
            return '{0} {1} {2}'.format(parts[0], self._separator, parts[1])
        elif len(parts) == 1 and parts[0] != '':
            return '{0} {1} ""'.format(parts[0], self._separator)