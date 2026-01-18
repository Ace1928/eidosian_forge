from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.snmp_server import (
def _compare_snmp_v3_auth_privacy(self, wattrib, hattrib, attrib):
    parsers = ['snmp_v3.trap_targets.authentication', 'snmp_v3.trap_targets.privacy', 'snmp_v3.users.authentication', 'snmp_v3.users.privacy']
    if attrib in ['trap_targets', 'users']:
        if attrib == 'users':
            primary_key = 'user'
        else:
            primary_key = 'address'
        for key, entry in iteritems(wattrib):
            if key != primary_key and key in ['authentication', 'privacy']:
                self.compare(parsers=parsers, want={'snmp_v3': {attrib: {key: entry, primary_key: wattrib[primary_key]}}}, have={'snmp_v3': {attrib: {key: hattrib.pop(key, {}), primary_key: wattrib[primary_key]}}})
        for key, entry in iteritems(hattrib):
            if key != primary_key and key in ['authentication', 'privacy']:
                self.compare(parsers=parsers, want={}, have={'snmp_v3': {attrib: {key: entry, primary_key: hattrib[primary_key]}}})