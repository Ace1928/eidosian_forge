from __future__ import absolute_import, division, print_function
import base64
import hashlib
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
class UserIPAClient(IPAClient):

    def __init__(self, module, host, port, protocol):
        super(UserIPAClient, self).__init__(module, host, port, protocol)

    def user_find(self, name):
        return self._post_json(method='user_find', name=None, item={'all': True, 'uid': name})

    def user_add(self, name, item):
        return self._post_json(method='user_add', name=name, item=item)

    def user_mod(self, name, item):
        return self._post_json(method='user_mod', name=name, item=item)

    def user_del(self, name):
        return self._post_json(method='user_del', name=name)

    def user_disable(self, name):
        return self._post_json(method='user_disable', name=name)

    def user_enable(self, name):
        return self._post_json(method='user_enable', name=name)