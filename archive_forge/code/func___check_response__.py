from __future__ import (absolute_import, division, print_function)
import re
import copy
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
def __check_response__(self, xml_str, xml_name):
    """Check if response message is already succeed"""
    if '<ok/>' not in xml_str:
        self.module.fail_json(msg='Error: %s failed.' % xml_name)