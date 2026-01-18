from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
class SudoRuleIPAClient(IPAClient):

    def __init__(self, module, host, port, protocol):
        super(SudoRuleIPAClient, self).__init__(module, host, port, protocol)

    def sudorule_find(self, name):
        return self._post_json(method='sudorule_find', name=None, item={'all': True, 'cn': name})

    def sudorule_add(self, name, item):
        return self._post_json(method='sudorule_add', name=name, item=item)

    def sudorule_add_runasuser(self, name, item):
        return self._post_json(method='sudorule_add_runasuser', name=name, item={'user': item})

    def sudorule_remove_runasuser(self, name, item):
        return self._post_json(method='sudorule_remove_runasuser', name=name, item={'user': item})

    def sudorule_mod(self, name, item):
        return self._post_json(method='sudorule_mod', name=name, item=item)

    def sudorule_del(self, name):
        return self._post_json(method='sudorule_del', name=name)

    def sudorule_add_option(self, name, item):
        return self._post_json(method='sudorule_add_option', name=name, item=item)

    def sudorule_add_option_ipasudoopt(self, name, item):
        return self.sudorule_add_option(name=name, item={'ipasudoopt': item})

    def sudorule_remove_option(self, name, item):
        return self._post_json(method='sudorule_remove_option', name=name, item=item)

    def sudorule_remove_option_ipasudoopt(self, name, item):
        return self.sudorule_remove_option(name=name, item={'ipasudoopt': item})

    def sudorule_add_host(self, name, item):
        return self._post_json(method='sudorule_add_host', name=name, item=item)

    def sudorule_add_host_host(self, name, item):
        return self.sudorule_add_host(name=name, item={'host': item})

    def sudorule_add_host_hostgroup(self, name, item):
        return self.sudorule_add_host(name=name, item={'hostgroup': item})

    def sudorule_remove_host(self, name, item):
        return self._post_json(method='sudorule_remove_host', name=name, item=item)

    def sudorule_remove_host_host(self, name, item):
        return self.sudorule_remove_host(name=name, item={'host': item})

    def sudorule_remove_host_hostgroup(self, name, item):
        return self.sudorule_remove_host(name=name, item={'hostgroup': item})

    def sudorule_add_allow_command(self, name, item):
        return self._post_json(method='sudorule_add_allow_command', name=name, item={'sudocmd': item})

    def sudorule_add_allow_command_group(self, name, item):
        return self._post_json(method='sudorule_add_allow_command', name=name, item={'sudocmdgroup': item})

    def sudorule_add_deny_command(self, name, item):
        return self._post_json(method='sudorule_add_deny_command', name=name, item={'sudocmd': item})

    def sudorule_add_deny_command_group(self, name, item):
        return self._post_json(method='sudorule_add_deny_command', name=name, item={'sudocmdgroup': item})

    def sudorule_remove_allow_command(self, name, item):
        return self._post_json(method='sudorule_remove_allow_command', name=name, item=item)

    def sudorule_add_user(self, name, item):
        return self._post_json(method='sudorule_add_user', name=name, item=item)

    def sudorule_add_user_user(self, name, item):
        return self.sudorule_add_user(name=name, item={'user': item})

    def sudorule_add_user_group(self, name, item):
        return self.sudorule_add_user(name=name, item={'group': item})

    def sudorule_remove_user(self, name, item):
        return self._post_json(method='sudorule_remove_user', name=name, item=item)

    def sudorule_remove_user_user(self, name, item):
        return self.sudorule_remove_user(name=name, item={'user': item})

    def sudorule_remove_user_group(self, name, item):
        return self.sudorule_remove_user(name=name, item={'group': item})