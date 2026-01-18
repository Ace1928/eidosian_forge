from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _icmp_match(self, rule, protocol, icmp_code, icmp_type):
    return protocol == 'icmp' and protocol == rule['protocol'] and (icmp_code == int(rule['icmpcode'])) and (icmp_type == int(rule['icmptype']))