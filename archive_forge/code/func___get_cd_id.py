from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def __get_cd_id(self):
    sds_service = self._connection.system_service().storage_domains_service()
    sds = sds_service.list(search='name="{0}"'.format(self.param('storage_domain') if self.param('storage_domain') else '*'))
    disks = self.__get_cds_from_sds(sds)
    if not disks:
        raise ValueError('Was not able to find disk with name or id "{0}".'.format(self.param('cd_iso')))
    if len(disks) > 1:
        raise ValueError('Found mutiple disks with same name "{0}" please use                 disk ID in "cd_iso" to specify which disk should be used.'.format(self.param('cd_iso')))
    return disks[0].id