from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class DatacentersModule(BaseModule):

    def __get_major(self, full_version):
        if full_version is None:
            return None
        if isinstance(full_version, otypes.Version):
            return full_version.major
        return int(full_version.split('.')[0])

    def __get_minor(self, full_version):
        if full_version is None:
            return None
        if isinstance(full_version, otypes.Version):
            return full_version.minor
        return int(full_version.split('.')[1])

    def _get_mac_pool(self):
        mac_pool = None
        if self._module.params.get('mac_pool'):
            mac_pool = search_by_name(self._connection.system_service().mac_pools_service(), self._module.params.get('mac_pool'))
        return mac_pool

    def build_entity(self):
        return otypes.DataCenter(name=self._module.params['name'], id=self._module.params['id'], comment=self._module.params['comment'], description=self._module.params['description'], mac_pool=otypes.MacPool(id=getattr(self._get_mac_pool(), 'id', None)) if self._module.params.get('mac_pool') else None, quota_mode=otypes.QuotaModeType(self._module.params['quota_mode']) if self._module.params['quota_mode'] else None, local=self._module.params['local'], version=otypes.Version(major=self.__get_major(self._module.params['compatibility_version']), minor=self.__get_minor(self._module.params['compatibility_version'])) if self._module.params['compatibility_version'] else None)

    def update_check(self, entity):
        minor = self.__get_minor(self._module.params.get('compatibility_version'))
        major = self.__get_major(self._module.params.get('compatibility_version'))
        return equal(getattr(self._get_mac_pool(), 'id', None), getattr(entity.mac_pool, 'id', None)) and equal(self._module.params.get('comment'), entity.comment) and equal(self._module.params.get('description'), entity.description) and equal(self._module.params.get('name'), entity.name) and equal(self._module.params.get('quota_mode'), str(entity.quota_mode)) and equal(self._module.params.get('local'), entity.local) and equal(minor, self.__get_minor(entity.version)) and equal(major, self.__get_major(entity.version))