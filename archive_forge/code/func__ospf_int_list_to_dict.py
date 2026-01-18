from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.ospf_interfaces import (
def _ospf_int_list_to_dict(self, entry):
    for name, family in iteritems(entry):
        if 'address_family' in family:
            addr_dict = {}
            for entry in family.get('address_family', []):
                addr_dict.update({entry['afi']: entry})
            family['address_family'] = addr_dict
            self._ospf_int_list_to_dict(family['address_family'])