from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
class GroupIPAClient(IPAClient):

    def __init__(self, module, host, port, protocol):
        super(GroupIPAClient, self).__init__(module, host, port, protocol)

    def group_find(self, name):
        return self._post_json(method='group_find', name=None, item={'all': True, 'cn': name})

    def group_add(self, name, item):
        return self._post_json(method='group_add', name=name, item=item)

    def group_mod(self, name, item):
        return self._post_json(method='group_mod', name=name, item=item)

    def group_del(self, name):
        return self._post_json(method='group_del', name=name)

    def group_add_member(self, name, item):
        return self._post_json(method='group_add_member', name=name, item=item)

    def group_add_member_group(self, name, item):
        return self.group_add_member(name=name, item={'group': item})

    def group_add_member_user(self, name, item):
        return self.group_add_member(name=name, item={'user': item})

    def group_add_member_externaluser(self, name, item):
        return self.group_add_member(name=name, item={'ipaexternalmember': item})

    def group_remove_member(self, name, item):
        return self._post_json(method='group_remove_member', name=name, item=item)

    def group_remove_member_group(self, name, item):
        return self.group_remove_member(name=name, item={'group': item})

    def group_remove_member_user(self, name, item):
        return self.group_remove_member(name=name, item={'user': item})

    def group_remove_member_externaluser(self, name, item):
        return self.group_remove_member(name=name, item={'ipaexternalmember': item})