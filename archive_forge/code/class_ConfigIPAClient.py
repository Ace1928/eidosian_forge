from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
class ConfigIPAClient(IPAClient):

    def __init__(self, module, host, port, protocol):
        super(ConfigIPAClient, self).__init__(module, host, port, protocol)

    def config_show(self):
        return self._post_json(method='config_show', name=None)

    def config_mod(self, name, item):
        return self._post_json(method='config_mod', name=name, item=item)