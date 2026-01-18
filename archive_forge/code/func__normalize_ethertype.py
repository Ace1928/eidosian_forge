from __future__ import absolute_import, division, print_function
from ast import literal_eval
from ansible.module_utils._text import to_text
from ansible.module_utils.common.validation import check_required_arguments
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
@staticmethod
def _normalize_ethertype(rule):
    ethertype = rule.get('ethertype')
    if ethertype:
        if ethertype.get('value'):
            value = ethertype.pop('value')
            if value.startswith('0x'):
                value = ETHERTYPE_FORMAT.format(int(value, 16))
            else:
                value = ETHERTYPE_FORMAT.format(int(value, 10))
            if value in ethertype_value_to_protocol_map:
                ethertype[ethertype_value_to_protocol_map[value]] = True
            else:
                ethertype['value'] = value
        elif not next(iter(ethertype.values())):
            del rule['ethertype']