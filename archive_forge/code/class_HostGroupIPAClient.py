from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
class HostGroupIPAClient(IPAClient):

    def __init__(self, module, host, port, protocol):
        super(HostGroupIPAClient, self).__init__(module, host, port, protocol)

    def hostgroup_find(self, name):
        return self._post_json(method='hostgroup_find', name=None, item={'all': True, 'cn': name})

    def hostgroup_add(self, name, item):
        return self._post_json(method='hostgroup_add', name=name, item=item)

    def hostgroup_mod(self, name, item):
        return self._post_json(method='hostgroup_mod', name=name, item=item)

    def hostgroup_del(self, name):
        return self._post_json(method='hostgroup_del', name=name)

    def hostgroup_add_member(self, name, item):
        return self._post_json(method='hostgroup_add_member', name=name, item=item)

    def hostgroup_add_host(self, name, item):
        return self.hostgroup_add_member(name=name, item={'host': item})

    def hostgroup_add_hostgroup(self, name, item):
        return self.hostgroup_add_member(name=name, item={'hostgroup': item})

    def hostgroup_remove_member(self, name, item):
        return self._post_json(method='hostgroup_remove_member', name=name, item=item)

    def hostgroup_remove_host(self, name, item):
        return self.hostgroup_remove_member(name=name, item={'host': item})

    def hostgroup_remove_hostgroup(self, name, item):
        return self.hostgroup_remove_member(name=name, item={'hostgroup': item})