from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
def _convertlentomask_(self, masklen):
    """convert mask length to ip address mask, i.e. 24 to 255.255.255.0"""
    mask_int = ['0'] * 4
    length = int(masklen)
    if length > 32:
        self.module.fail_json(msg='IPv4 ipaddress mask length is invalid')
    if length < 8:
        mask_int[0] = str(int(255 << 8 - length % 8 & 255))
    if length >= 8:
        mask_int[0] = '255'
        mask_int[1] = str(int(255 << 16 - length % 16 & 255))
    if length >= 16:
        mask_int[1] = '255'
        mask_int[2] = str(int(255 << 24 - length % 24 & 255))
    if length >= 24:
        mask_int[2] = '255'
        mask_int[3] = str(int(255 << 32 - length % 32 & 255))
    if length == 32:
        mask_int[3] = '255'
    return '.'.join(mask_int)