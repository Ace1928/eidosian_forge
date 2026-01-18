from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_qoss_service(connection, dc_name):
    """
    Gets the qoss_service from the data_center provided

    :returns: ovirt.services.QossService or None
    """
    dcs_service = connection.system_service().data_centers_service()
    return dcs_service.data_center_service(get_id_by_name(dcs_service, dc_name)).qoss_service()