from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
class RoleIPAClient(IPAClient):

    def __init__(self, module, host, port, protocol):
        super(RoleIPAClient, self).__init__(module, host, port, protocol)

    def role_find(self, name):
        return self._post_json(method='role_find', name=None, item={'all': True, 'cn': name})

    def role_add(self, name, item):
        return self._post_json(method='role_add', name=name, item=item)

    def role_mod(self, name, item):
        return self._post_json(method='role_mod', name=name, item=item)

    def role_del(self, name):
        return self._post_json(method='role_del', name=name)

    def role_add_member(self, name, item):
        return self._post_json(method='role_add_member', name=name, item=item)

    def role_add_group(self, name, item):
        return self.role_add_member(name=name, item={'group': item})

    def role_add_host(self, name, item):
        return self.role_add_member(name=name, item={'host': item})

    def role_add_hostgroup(self, name, item):
        return self.role_add_member(name=name, item={'hostgroup': item})

    def role_add_service(self, name, item):
        return self.role_add_member(name=name, item={'service': item})

    def role_add_user(self, name, item):
        return self.role_add_member(name=name, item={'user': item})

    def role_remove_member(self, name, item):
        return self._post_json(method='role_remove_member', name=name, item=item)

    def role_remove_group(self, name, item):
        return self.role_remove_member(name=name, item={'group': item})

    def role_remove_host(self, name, item):
        return self.role_remove_member(name=name, item={'host': item})

    def role_remove_hostgroup(self, name, item):
        return self.role_remove_member(name=name, item={'hostgroup': item})

    def role_remove_service(self, name, item):
        return self.role_remove_member(name=name, item={'service': item})

    def role_remove_user(self, name, item):
        return self.role_remove_member(name=name, item={'user': item})

    def role_add_privilege(self, name, item):
        return self._post_json(method='role_add_privilege', name=name, item={'privilege': item})

    def role_remove_privilege(self, name, item):
        return self._post_json(method='role_remove_privilege', name=name, item={'privilege': item})