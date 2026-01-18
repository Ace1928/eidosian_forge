from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.facts.facts import Facts
from ansible_collections.cisco.asa.plugins.module_utils.network.asa.rm_templates.ogs import (
def _icmp_object_compare(self, want, have):
    icmp_obj = 'icmp_type'
    for name, entry in iteritems(want):
        h_item = have.pop(name, {})
        if entry != h_item and name != 'object_type' and entry[icmp_obj].get('icmp_object'):
            if h_item and entry.get('group_object'):
                self.addcmd(entry, 'og_name', False)
                self._add_group_object_cmd(entry, h_item)
                continue
            if h_item:
                self._add_object_cmd(entry, h_item, icmp_obj, ['icmp_type'])
            else:
                self.addcmd(entry, 'og_name', False)
                self.compare(['description'], entry, h_item)
            if entry.get('group_object'):
                self._add_group_object_cmd(entry, h_item)
                continue
            if self.state in ('overridden', 'replaced') and h_item:
                self.compare(['icmp_type'], {}, h_item)
            if h_item and h_item[icmp_obj].get('icmp_object'):
                li_diff = self.get_list_diff(entry, h_item, icmp_obj, 'icmp_object')
            else:
                li_diff = entry[icmp_obj].get('icmp_object')
            entry[icmp_obj]['icmp_object'] = li_diff
            self.addcmd(entry, 'icmp_type', False)
    self.check_for_have_and_overidden(have)