from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_qos(self):
    """
        Gets the QoS entry if exists

        :return: otypes.QoS or None
        """
    dc_name = self._module.params.get('data_center')
    dcs_service = self._connection.system_service().data_centers_service()
    qos_service = dcs_service.data_center_service(get_id_by_name(dcs_service, dc_name)).qoss_service()
    return get_entity(qos_service.qos_service(get_id_by_name(qos_service, self._module.params.get('qos'))))