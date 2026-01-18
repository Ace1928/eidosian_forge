from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.service.service import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.service import (
class ServiceFacts(object):
    """The ios service facts class"""

    def __init__(self, module, subspec='config', options='options'):
        self._module = module
        self.argument_spec = ServiceArgs.argument_spec

    def get_service_data(self, connection):
        return connection.get('show running-config all | section ^service ')

    def populate_facts(self, connection, ansible_facts, data=None):
        """Populate the facts for Service network resource

        :param connection: the device connection
        :param ansible_facts: Facts dictionary
        :param data: previously collected conf

        :rtype: dictionary
        :returns: facts
        """
        facts = {}
        objs = []
        params = {}
        if not data:
            data = self.get_service_data(connection)
        service_parser = ServiceTemplate(lines=data.splitlines(), module=self._module)
        objs = service_parser.parse()
        ansible_facts['ansible_network_resources'].pop('service', None)
        params = utils.remove_empties(service_parser.validate_config(self.argument_spec, {'config': objs}, redact=True))
        facts['service'] = params.get('config', {})
        ansible_facts['ansible_network_resources'].update(facts)
        return ansible_facts