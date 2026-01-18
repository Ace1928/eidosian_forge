from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
class ServiceIPAClient(IPAClient):

    def __init__(self, module, host, port, protocol):
        super(ServiceIPAClient, self).__init__(module, host, port, protocol)

    def service_find(self, name):
        return self._post_json(method='service_find', name=None, item={'all': True, 'krbcanonicalname': name})

    def service_add(self, name, service):
        return self._post_json(method='service_add', name=name, item=service)

    def service_mod(self, name, service):
        return self._post_json(method='service_mod', name=name, item=service)

    def service_del(self, name):
        return self._post_json(method='service_del', name=name)

    def service_disable(self, name):
        return self._post_json(method='service_disable', name=name)

    def service_add_host(self, name, item):
        return self._post_json(method='service_add_host', name=name, item={'host': item})

    def service_remove_host(self, name, item):
        return self._post_json(method='service_remove_host', name=name, item={'host': item})