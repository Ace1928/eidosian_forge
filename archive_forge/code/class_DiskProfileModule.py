from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class DiskProfileModule(BaseModule):

    def _get_qos(self):
        """
        Gets the QoS entry if exists

        :return: otypes.QoS or None
        """
        dc_name = self._module.params.get('data_center')
        dcs_service = self._connection.system_service().data_centers_service()
        qos_service = dcs_service.data_center_service(get_id_by_name(dcs_service, dc_name)).qoss_service()
        return get_entity(qos_service.qos_service(get_id_by_name(qos_service, self._module.params.get('qos'))))

    def _get_storage_domain(self):
        """
        Gets the storage domain

        :return: otypes.StorageDomain or None
        """
        storage_domain_name = self._module.params.get('storage_domain')
        storage_domains_service = self._connection.system_service().storage_domains_service()
        return get_entity(storage_domains_service.storage_domain_service(get_id_by_name(storage_domains_service, storage_domain_name)))

    def build_entity(self):
        """
        Abstract method from BaseModule called from create() and remove()

        Builds the disk profile from the given params

        :return: otypes.DiskProfile
        """
        qos = self._get_qos()
        storage_domain = self._get_storage_domain()
        if qos is None:
            raise Exception('The qos: {0} does not exist in data center: {1}'.format(self._module.params.get('qos'), self._module.params.get('data_center')))
        if storage_domain is None:
            raise Exception('The storage domain: {0} does not exist.'.format(self._module.params.get('storage_domain')))
        return otypes.DiskProfile(name=self._module.params.get('name') if self._module.params.get('name') else None, id=self._module.params.get('id') if self._module.params.get('id') else None, comment=self._module.params.get('comment'), description=self._module.params.get('description'), qos=qos, storage_domain=storage_domain)