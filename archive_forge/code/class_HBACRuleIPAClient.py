from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
class HBACRuleIPAClient(IPAClient):

    def __init__(self, module, host, port, protocol):
        super(HBACRuleIPAClient, self).__init__(module, host, port, protocol)

    def hbacrule_find(self, name):
        return self._post_json(method='hbacrule_find', name=None, item={'all': True, 'cn': name})

    def hbacrule_add(self, name, item):
        return self._post_json(method='hbacrule_add', name=name, item=item)

    def hbacrule_mod(self, name, item):
        return self._post_json(method='hbacrule_mod', name=name, item=item)

    def hbacrule_del(self, name):
        return self._post_json(method='hbacrule_del', name=name)

    def hbacrule_add_host(self, name, item):
        return self._post_json(method='hbacrule_add_host', name=name, item=item)

    def hbacrule_remove_host(self, name, item):
        return self._post_json(method='hbacrule_remove_host', name=name, item=item)

    def hbacrule_add_service(self, name, item):
        return self._post_json(method='hbacrule_add_service', name=name, item=item)

    def hbacrule_remove_service(self, name, item):
        return self._post_json(method='hbacrule_remove_service', name=name, item=item)

    def hbacrule_add_user(self, name, item):
        return self._post_json(method='hbacrule_add_user', name=name, item=item)

    def hbacrule_remove_user(self, name, item):
        return self._post_json(method='hbacrule_remove_user', name=name, item=item)

    def hbacrule_add_sourcehost(self, name, item):
        return self._post_json(method='hbacrule_add_sourcehost', name=name, item=item)

    def hbacrule_remove_sourcehost(self, name, item):
        return self._post_json(method='hbacrule_remove_sourcehost', name=name, item=item)