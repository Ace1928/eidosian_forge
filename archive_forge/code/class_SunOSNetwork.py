from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.facts.network.base import NetworkCollector
from ansible.module_utils.facts.network.generic_bsd import GenericBsdIfconfigNetwork
class SunOSNetwork(GenericBsdIfconfigNetwork):
    """
    This is the SunOS Network Class.
    It uses the GenericBsdIfconfigNetwork.

    Solaris can have different FLAGS and MTU for IPv4 and IPv6 on the same interface
    so these facts have been moved inside the 'ipv4' and 'ipv6' lists.
    """
    platform = 'SunOS'

    def get_interfaces_info(self, ifconfig_path):
        interfaces = {}
        current_if = {}
        ips = dict(all_ipv4_addresses=[], all_ipv6_addresses=[])
        rc, out, err = self.module.run_command([ifconfig_path, '-a'])
        for line in out.splitlines():
            if line:
                words = line.split()
                if re.match('^\\S', line) and len(words) > 3:
                    current_if = self.parse_interface_line(words, current_if, interfaces)
                    interfaces[current_if['device']] = current_if
                elif words[0].startswith('options='):
                    self.parse_options_line(words, current_if, ips)
                elif words[0] == 'nd6':
                    self.parse_nd6_line(words, current_if, ips)
                elif words[0] == 'ether':
                    self.parse_ether_line(words, current_if, ips)
                elif words[0] == 'media:':
                    self.parse_media_line(words, current_if, ips)
                elif words[0] == 'status:':
                    self.parse_status_line(words, current_if, ips)
                elif words[0] == 'lladdr':
                    self.parse_lladdr_line(words, current_if, ips)
                elif words[0] == 'inet':
                    self.parse_inet_line(words, current_if, ips)
                elif words[0] == 'inet6':
                    self.parse_inet6_line(words, current_if, ips)
                else:
                    self.parse_unknown_line(words, current_if, ips)
        for iface in interfaces:
            for v in ('ipv4', 'ipv6'):
                combined_facts = {}
                for facts in interfaces[iface][v]:
                    combined_facts.update(facts)
                if len(combined_facts.keys()) > 0:
                    interfaces[iface][v] = [combined_facts]
        return (interfaces, ips)

    def parse_interface_line(self, words, current_if, interfaces):
        device = words[0][0:-1]
        if device not in interfaces:
            current_if = {'device': device, 'ipv4': [], 'ipv6': [], 'type': 'unknown'}
        else:
            current_if = interfaces[device]
        flags = self.get_options(words[1])
        v = 'ipv4'
        if 'IPv6' in flags:
            v = 'ipv6'
        if 'LOOPBACK' in flags:
            current_if['type'] = 'loopback'
        current_if[v].append({'flags': flags, 'mtu': words[3]})
        current_if['macaddress'] = 'unknown'
        return current_if

    def parse_ether_line(self, words, current_if, ips):
        macaddress = ''
        for octet in words[1].split(':'):
            octet = ('0' + octet)[-2:None]
            macaddress += octet + ':'
        current_if['macaddress'] = macaddress[0:-1]