from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _ah_esp_gre_match(self, rule, protocol):
    return protocol in ['ah', 'esp', 'gre'] and protocol == rule['protocol']