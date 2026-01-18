from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class GroupsModule(BaseModule):

    def build_entity(self):
        return otypes.Group(domain=otypes.Domain(name=self._module.params['authz_name']), name=self._module.params['name'], namespace=self._module.params['namespace'])