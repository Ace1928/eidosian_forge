from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
class SudoCmdGroupIPAClient(IPAClient):

    def __init__(self, module, host, port, protocol):
        super(SudoCmdGroupIPAClient, self).__init__(module, host, port, protocol)

    def sudocmdgroup_find(self, name):
        return self._post_json(method='sudocmdgroup_find', name=None, item={'all': True, 'cn': name})

    def sudocmdgroup_add(self, name, item):
        return self._post_json(method='sudocmdgroup_add', name=name, item=item)

    def sudocmdgroup_mod(self, name, item):
        return self._post_json(method='sudocmdgroup_mod', name=name, item=item)

    def sudocmdgroup_del(self, name):
        return self._post_json(method='sudocmdgroup_del', name=name)

    def sudocmdgroup_add_member(self, name, item):
        return self._post_json(method='sudocmdgroup_add_member', name=name, item=item)

    def sudocmdgroup_add_member_sudocmd(self, name, item):
        return self.sudocmdgroup_add_member(name=name, item={'sudocmd': item})

    def sudocmdgroup_remove_member(self, name, item):
        return self._post_json(method='sudocmdgroup_remove_member', name=name, item=item)

    def sudocmdgroup_remove_member_sudocmd(self, name, item):
        return self.sudocmdgroup_remove_member(name=name, item={'sudocmd': item})