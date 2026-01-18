from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.evpn_global.evpn_global import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.evpn_global import (
class Evpn_globalFacts(object):
    """The ios evpn_global facts class"""

    def __init__(self, module):
        self._module = module
        self.argument_spec = Evpn_globalArgs.argument_spec

    def get_evpn_global_data(self, connection):
        return connection.get('show running-config | section ^l2vpn evpn$')

    def populate_facts(self, connection, ansible_facts, data=None):
        """Populate the facts for Evpn_global network resource

        :param connection: the device connection
        :param ansible_facts: Facts dictionary
        :param data: previously collected conf

        :rtype: dictionary
        :returns: facts
        """
        facts = {}
        objs = []
        if not data:
            data = self.get_evpn_global_data(connection)
        evpn_global_parser = Evpn_globalTemplate(lines=data.splitlines(), module=self._module)
        objs = evpn_global_parser.parse()
        obj = utils.remove_empties(objs)
        ansible_facts['ansible_network_resources'].pop('evpn_global', None)
        params = utils.remove_empties(evpn_global_parser.validate_config(self.argument_spec, {'config': obj}, redact=True))
        facts['evpn_global'] = params.get('config', {})
        ansible_facts['ansible_network_resources'].update(facts)
        return ansible_facts