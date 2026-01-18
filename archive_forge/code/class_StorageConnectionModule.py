from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class StorageConnectionModule(BaseModule):

    def build_entity(self):
        return otypes.StorageConnection(address=self.param('address'), path=self.param('path'), nfs_version=otypes.NfsVersion(self.param('nfs_version')) if self.param('nfs_version') is not None else None, nfs_timeo=self.param('nfs_timeout'), nfs_retrans=self.param('nfs_retrans'), mount_options=self.param('mount_options'), password=self.param('password'), username=self.param('username'), port=self.param('port'), target=self.param('target'), type=otypes.StorageType(self.param('type')) if self.param('type') is not None else None, vfs_type=self.param('vfs_type'))

    def _get_storage_domain_service(self):
        sds_service = self._connection.system_service().storage_domains_service()
        sd = search_by_name(sds_service, self.param('storage'))
        if sd is None:
            raise Exception("Storage '%s' was not found." % self.param('storage'))
        return (sd, sds_service.storage_domain_service(sd.id))

    def post_present(self, entity_id):
        if self.param('storage'):
            sd, sd_service = self._get_storage_domain_service()
            if entity_id not in [sd_conn.id for sd_conn in self._connection.follow_link(sd.storage_connections)]:
                scs_service = sd_service.storage_connections_service()
                if not self._module.check_mode:
                    scs_service.add(connection=otypes.StorageConnection(id=entity_id))
                self.changed = True

    def pre_remove(self, entity):
        if self.param('storage'):
            sd, sd_service = self._get_storage_domain_service()
            if entity in [sd_conn.id for sd_conn in self._connection.follow_link(sd.storage_connections)]:
                scs_service = sd_service.storage_connections_service()
                sc_service = scs_service.connection_service(entity)
                if not self._module.check_mode:
                    sc_service.remove()
                self.changed = True

    def update_check(self, entity):
        return equal(self.param('address'), entity.address) and equal(self.param('path'), entity.path) and equal(self.param('nfs_version'), str(entity.nfs_version)) and equal(self.param('nfs_timeout'), entity.nfs_timeo) and equal(self.param('nfs_retrans'), entity.nfs_retrans) and equal(self.param('mount_options'), entity.mount_options) and equal(self.param('username'), entity.username) and equal(self.param('port'), entity.port) and equal(self.param('target'), entity.target) and equal(self.param('type'), str(entity.type)) and equal(self.param('vfs_type'), entity.vfs_type)