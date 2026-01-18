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
def _convert_port_dict_to_payload_format(port_dict):
    payload = None
    if port_dict:
        if port_dict.get('eq') is not None:
            payload = port_dict['eq']
        elif port_dict.get('lt') is not None:
            payload = '{0}..{1}'.format(L4_PORT_START, port_dict['lt'])
        elif port_dict.get('gt') is not None:
            payload = '{0}..{1}'.format(port_dict['gt'], L4_PORT_END)
        elif port_dict.get('range'):
            payload = '{0}..{1}'.format(port_dict['range']['begin'], port_dict['range']['end'])
    return payload