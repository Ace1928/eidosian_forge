from __future__ import (absolute_import, division, print_function)
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
from ansible.module_utils.basic import AnsibleModule
import traceback
class RoleModule(BaseModule):

    def build_entity(self):
        if 'login' not in self.param('permits'):
            self.param('permits').append('login')
        all_permits = self.get_all_permits()
        return otypes.Role(id=self.param('id'), name=self.param('name'), administrative=self.param('administrative') if self.param('administrative') is not None else None, permits=[otypes.Permit(id=all_permits.get(new_permit)) for new_permit in self.param('permits')] if self.param('permits') else None, description=self.param('description') if self.param('administrative') else None)

    def get_all_permits(self):
        return dict(((permit.name, permit.id) for permit in self._connection.system_service().cluster_levels_service().level_service('4.3').get().permits))

    def update_check(self, entity):

        def check_permits():
            if self.param('permits'):
                if 'login' not in self.param('permits'):
                    self.param('permits').append('login')
                permits_service = self._service.service(entity.id).permits_service()
                current = [er.name for er in permits_service.list()]
                passed = self.param('permits')
                if not sorted(current) == sorted(passed):
                    if self._module.check_mode:
                        return False
                    for permit in permits_service.list():
                        permits_service.permit_service(permit.id).remove()
                    all_permits = self.get_all_permits()
                    for new_permit in passed:
                        permits_service.add(otypes.Permit(id=all_permits.get(new_permit)))
                    return False
            return True
        return check_permits() and equal(self.param('administrative'), entity.administrative) and equal(self.param('description'), entity.description)