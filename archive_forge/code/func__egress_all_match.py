from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _egress_all_match(self, rule, protocol, fw_type):
    return protocol in ['all'] and protocol == rule['protocol'] and (fw_type == 'egress')