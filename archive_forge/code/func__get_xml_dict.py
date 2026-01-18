from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.ospfv2.ospfv2 import (
def _get_xml_dict(self, xml_root):
    if not HAS_XMLTODICT:
        self._module.fail_json(msg=missing_required_lib('xmltodict'))
    xml_dict = xmltodict.parse(etree.tostring(xml_root), dict_constructor=dict)
    return xml_dict