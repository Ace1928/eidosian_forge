from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class EventsModule(BaseModule):

    def build_entity(self):
        correlation_id = None
        if self._module.params['correlation_id'] is not None:
            correlation_id = self._module.params['correlation_id']
        elif self._connection._headers.get('correlation-id') is not None:
            correlation_id = self._connection._headers.get('correlation-id')
        return otypes.Event(description=self._module.params['description'], severity=otypes.LogSeverity(self._module.params['severity']), origin=self._module.params['origin'], custom_id=self._module.params['custom_id'], id=self._module.params['id'], correlation_id=correlation_id, cluster=otypes.Cluster(id=self._module.params['cluster']) if self._module.params['cluster'] is not None else None, data_center=otypes.DataCenter(id=self._module.params['data_center']) if self._module.params['data_center'] is not None else None, host=otypes.Host(id=self._module.params['host']) if self._module.params['host'] is not None else None, storage_domain=otypes.StorageDomain(id=self._module.params['storage_domain']) if self._module.params['storage_domain'] is not None else None, template=otypes.Template(id=self._module.params['template']) if self._module.params['template'] is not None else None, user=otypes.User(id=self._module.params['user']) if self._module.params['user'] is not None else None, vm=otypes.Vm(id=self._module.params['vm']) if self._module.params['vm'] is not None else None)