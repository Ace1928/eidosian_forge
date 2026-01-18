from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.snmp_server import (
def _compare_snmp_v3(self, want, have):
    parsers = ['snmp_v3.groups', 'snmp_v3.trap_targets.port', 'snmp_v3.trap_targets.protocol', 'snmp_v3.trap_targets.type', 'snmp_v3.trap_targets.user', 'snmp_v3.users.group', 'snmp_v3.users.mode', 'snmp_v3.views', 'snmp_v3.engine_id']
    attribute_dict = {'groups': 'group', 'trap_targets': 'address', 'users': 'user', 'views': 'view'}
    wdict = get_from_dict(want, 'snmp_v3') or {}
    hdict = get_from_dict(have, 'snmp_v3') or {}
    for attrib in attribute_dict.keys():
        wattrib = get_from_dict(wdict, attrib) or {}
        hattrib = get_from_dict(hdict, attrib) or {}
        for key, entry in iteritems(wattrib):
            self._compare_snmp_v3_auth_privacy(entry, hattrib.get(key, {}), attrib)
            for k, v in iteritems(entry):
                if k != attribute_dict[attrib]:
                    h = {}
                    if hattrib.get(key):
                        h = {'snmp_v3': {attrib: {k: hattrib[key].pop(k, ''), attribute_dict[attrib]: hattrib[key][attribute_dict[attrib]]}}}
                    self.compare(parsers=parsers, want={'snmp_v3': {attrib: {k: v, attribute_dict[attrib]: entry[attribute_dict[attrib]]}}}, have=h)
        for key, entry in iteritems(hattrib):
            self._compare_snmp_v3_auth_privacy({}, entry, attrib)
            self.compare(parsers=parsers, want={}, have={'snmp_v3': {attrib: entry}})
        hdict.pop(attrib, {})
    for key, entry in iteritems(wdict):
        self.compare(parsers='snmp_v3.engine_id', want={'snmp_v3': {key: entry}}, have={'snmp_v3': {key: hdict.pop(key, {})}})
    for key, entry in iteritems(hdict):
        self.compare(parsers=parsers, want={}, have={'snmp_v3': {key: entry}})