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
def _convert_config_list_to_dict(config_list):
    config_dict = {}
    for config in config_list:
        acl_type = config['address_family']
        config_dict[acl_type] = {}
        if config.get('acls'):
            for acl in config['acls']:
                acl_name = acl['name']
                config_dict[acl_type][acl_name] = {}
                config_dict[acl_type][acl_name]['remark'] = acl.get('remark')
                config_dict[acl_type][acl_name]['rules'] = {}
                if acl.get('rules'):
                    for rule in acl['rules']:
                        config_dict[acl_type][acl_name]['rules'][rule['sequence_num']] = rule
    return config_dict