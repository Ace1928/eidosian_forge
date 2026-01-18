from __future__ import (absolute_import, division, print_function)
import re
import copy
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def __string_to_lowercase__(self):
    """Convert string to lowercase"""
    if self.route_distinguisher:
        self.route_distinguisher = self.route_distinguisher.lower()
    if self.vpn_target_export:
        for index, ele in enumerate(self.vpn_target_export):
            self.vpn_target_export[index] = ele.lower()
    if self.vpn_target_import:
        for index, ele in enumerate(self.vpn_target_import):
            self.vpn_target_import[index] = ele.lower()
    if self.vpn_target_both:
        for index, ele in enumerate(self.vpn_target_both):
            self.vpn_target_both[index] = ele.lower()