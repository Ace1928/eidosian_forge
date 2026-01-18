from __future__ import (absolute_import, division, print_function)
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_all_data_with_pagination
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def _check_duplicate_device_id(device_id_list, service_tag_dict):
    """If service_tag is duplicate of device_id, then updates the message as Duplicate report
    :arg1: device_id_list : list of device_id
    :arg2: service_tag_id_dict: dictionary of device_id to service tag map"""
    if device_id_list:
        device_id_represents_int = [int(device_id) for device_id in device_id_list if device_id and is_int(device_id)]
        common_val = list(set(device_id_represents_int) & set(service_tag_dict.keys()))
        for device_id in common_val:
            device_fact_error_report.update({service_tag_dict[device_id]: 'Duplicate report of device_id: {0}'.format(device_id)})
            del service_tag_dict[device_id]