from __future__ import (absolute_import, division, print_function)
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_all_data_with_pagination
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def _get_device_id_from_service_tags(service_tags, rest_obj):
    """
    Get device ids from device service tag
    Returns :dict : device_id to service_tag map
    :arg service_tags: service tag
    :arg rest_obj: RestOME class object in case of request with session.
    :returns: dict eg: {1345:"MXL1245"}
    """
    device_list = rest_obj.get_all_report_details(DEVICE_RESOURCE_COLLECTION[DEVICE_LIST]['resource'])['report_list']
    service_tag_dict = {}
    for item in device_list:
        if item['DeviceServiceTag'] in service_tags:
            service_tag_dict.update({item['Id']: item['DeviceServiceTag']})
    available_service_tags = service_tag_dict.values()
    missing_service_tags = list(set(service_tags) - set(available_service_tags))
    update_device_details_with_filtering(missing_service_tags, service_tag_dict, rest_obj)
    device_fact_error_report.update(dict(((tag, DESC_HTTP_ERROR) for tag in missing_service_tags)))
    return service_tag_dict