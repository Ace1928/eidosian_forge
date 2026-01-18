from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
class DNSZoneIPAClient(IPAClient):

    def __init__(self, module, host, port, protocol):
        super(DNSZoneIPAClient, self).__init__(module, host, port, protocol)

    def dnszone_find(self, zone_name, details=None):
        items = {'all': 'true', 'idnsname': zone_name}
        if details is not None:
            items.update(details)
        return self._post_json(method='dnszone_find', name=zone_name, item=items)

    def dnszone_add(self, zone_name=None, details=None):
        items = {}
        if details is not None:
            items.update(details)
        return self._post_json(method='dnszone_add', name=zone_name, item=items)

    def dnszone_mod(self, zone_name=None, details=None):
        items = {}
        if details is not None:
            items.update(details)
        return self._post_json(method='dnszone_mod', name=zone_name, item=items)

    def dnszone_del(self, zone_name=None, record_name=None, details=None):
        return self._post_json(method='dnszone_del', name=zone_name, item={})