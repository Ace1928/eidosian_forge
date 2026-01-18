from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
class VaultIPAClient(IPAClient):

    def __init__(self, module, host, port, protocol):
        super(VaultIPAClient, self).__init__(module, host, port, protocol)

    def vault_find(self, name):
        return self._post_json(method='vault_find', name=None, item={'all': True, 'cn': name})

    def vault_add_internal(self, name, item):
        return self._post_json(method='vault_add_internal', name=name, item=item)

    def vault_mod_internal(self, name, item):
        return self._post_json(method='vault_mod_internal', name=name, item=item)

    def vault_del(self, name):
        return self._post_json(method='vault_del', name=name)