from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.facts.network.base import NetworkCollector
from ansible.module_utils.facts.network.generic_bsd import GenericBsdIfconfigNetwork
class NetBSDNetwork(GenericBsdIfconfigNetwork):
    """
    This is the NetBSD Network Class.
    It uses the GenericBsdIfconfigNetwork
    """
    platform = 'NetBSD'

    def parse_media_line(self, words, current_if, ips):
        current_if['media'] = words[1]
        if len(words) > 2:
            current_if['media_type'] = words[2]
        if len(words) > 3:
            current_if['media_options'] = words[3].split(',')