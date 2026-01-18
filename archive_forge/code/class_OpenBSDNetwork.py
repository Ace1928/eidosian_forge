from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.facts.network.base import NetworkCollector
from ansible.module_utils.facts.network.generic_bsd import GenericBsdIfconfigNetwork
class OpenBSDNetwork(GenericBsdIfconfigNetwork):
    """
    This is the OpenBSD Network Class.
    It uses the GenericBsdIfconfigNetwork.
    """
    platform = 'OpenBSD'

    def get_interfaces_info(self, ifconfig_path, ifconfig_options='-aA'):
        return super(OpenBSDNetwork, self).get_interfaces_info(ifconfig_path, ifconfig_options)

    def parse_lladdr_line(self, words, current_if, ips):
        current_if['macaddress'] = words[1]
        current_if['type'] = 'ether'