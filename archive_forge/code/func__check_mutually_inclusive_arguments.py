from __future__ import (absolute_import, division, print_function)
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_all_data_with_pagination
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def _check_mutually_inclusive_arguments(val, module_params, required_args):
    """"
     Throws error if arguments detailed_inventory, subsystem_health
     not exists with qualifier device_id or device_service_tag"""
    system_query_options_param = module_params.get('system_query_options')
    if system_query_options_param is None or (system_query_options_param is not None and (not any((system_query_options_param.get(qualifier) for qualifier in required_args)))):
        raise ValueError('One of the following {0} is required for {1}'.format(required_args, val))