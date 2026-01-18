from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.vxlan_vtep.vxlan_vtep import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.vxlan_vtep import (
class Vxlan_vtepFacts(object):
    """The ios vxlan_vtep facts class"""

    def __init__(self, module):
        self._module = module
        self.argument_spec = Vxlan_vtepArgs.argument_spec

    def get_vxlan_vtep_data(self, connection):
        return connection.get('show running-config | section ^interface nve')

    def populate_facts(self, connection, ansible_facts, data=None):
        """Populate the facts for Vxlan_vtep network resource

        :param connection: the device connection
        :param ansible_facts: Facts dictionary
        :param data: previously collected conf

        :rtype: dictionary
        :returns: facts
        """
        facts = {}
        objs = []
        if not data:
            data = self.get_vxlan_vtep_data(connection)
        vxlan_vtep_parser = Vxlan_vtepTemplate(lines=data.splitlines(), module=self._module)
        objs = list(vxlan_vtep_parser.parse().values())
        ansible_facts['ansible_network_resources'].pop('vxlan_vtep', None)
        params = utils.remove_empties(vxlan_vtep_parser.validate_config(self.argument_spec, {'config': objs}, redact=True))
        facts['vxlan_vtep'] = params.get('config', {})
        ansible_facts['ansible_network_resources'].update(facts)
        return ansible_facts