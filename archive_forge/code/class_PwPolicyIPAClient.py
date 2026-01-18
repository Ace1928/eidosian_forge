from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
class PwPolicyIPAClient(IPAClient):
    """The global policy will be selected when `name` is `None`"""

    def __init__(self, module, host, port, protocol):
        super(PwPolicyIPAClient, self).__init__(module, host, port, protocol)

    def pwpolicy_find(self, name):
        if name is None:
            name = 'global_policy'
        return self._post_json(method='pwpolicy_find', name=None, item={'all': True, 'cn': name})

    def pwpolicy_add(self, name, item):
        return self._post_json(method='pwpolicy_add', name=name, item=item)

    def pwpolicy_mod(self, name, item):
        return self._post_json(method='pwpolicy_mod', name=name, item=item)

    def pwpolicy_del(self, name):
        return self._post_json(method='pwpolicy_del', name=name)