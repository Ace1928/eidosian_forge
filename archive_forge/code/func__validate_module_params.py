from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _validate_module_params(clc, module):
    """
        Validate the module params, and lookup default values.
        :param clc: clc-sdk instance to use
        :param module: module to validate
        :return: dictionary of validated params
        """
    params = module.params
    datacenter = ClcServer._find_datacenter(clc, module)
    ClcServer._validate_types(module)
    ClcServer._validate_name(module)
    params['alias'] = ClcServer._find_alias(clc, module)
    params['cpu'] = ClcServer._find_cpu(clc, module)
    params['memory'] = ClcServer._find_memory(clc, module)
    params['description'] = ClcServer._find_description(module)
    params['ttl'] = ClcServer._find_ttl(clc, module)
    params['template'] = ClcServer._find_template_id(module, datacenter)
    params['group'] = ClcServer._find_group(module, datacenter).id
    params['network_id'] = ClcServer._find_network_id(module, datacenter)
    params['anti_affinity_policy_id'] = ClcServer._find_aa_policy_id(clc, module)
    params['alert_policy_id'] = ClcServer._find_alert_policy_id(clc, module)
    return params