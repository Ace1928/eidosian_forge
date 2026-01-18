from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
class SudoCmdIPAClient(IPAClient):

    def __init__(self, module, host, port, protocol):
        super(SudoCmdIPAClient, self).__init__(module, host, port, protocol)

    def sudocmd_find(self, name):
        return self._post_json(method='sudocmd_find', name=None, item={'all': True, 'sudocmd': name})

    def sudocmd_add(self, name, item):
        return self._post_json(method='sudocmd_add', name=name, item=item)

    def sudocmd_mod(self, name, item):
        return self._post_json(method='sudocmd_mod', name=name, item=item)

    def sudocmd_del(self, name):
        return self._post_json(method='sudocmd_del', name=name)