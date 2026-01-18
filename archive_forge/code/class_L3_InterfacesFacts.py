from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.l3_interfaces.l3_interfaces import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.l3_interfaces import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.utils.utils import (
class L3_InterfacesFacts(object):
    """The ios l3 interfaces fact class"""

    def __init__(self, module, subspec='config', options='options'):
        self._module = module
        self.argument_spec = L3_interfacesArgs.argument_spec

    def get_l3_interfaces_data(self, connection):
        return connection.get('show running-config | section ^interface')

    def populate_facts(self, connection, ansible_facts, data=None):
        """Populate the facts for l3 interfaces
        :param connection: the device connection
        :param ansible_facts: Facts dictionary
        :param data: previously collected conf
        :rtype: dictionary
        :returns: facts
        """
        objs = []
        if not data:
            data = self.get_l3_interfaces_data(connection)
        l3_interfaces_parser = L3_interfacesTemplate(lines=data.splitlines())
        objs = l3_interfaces_parser.parse()
        objs = utils.remove_empties(objs)
        temp = []
        for k, v in iteritems(objs):
            if v.get('ipv4'):
                for each in v['ipv4']:
                    if each.get('netmask'):
                        cidr_val = netmask_to_cidr(each['netmask'])
                        each['address'] = each['address'].strip(' ') + '/' + cidr_val
                        del each['netmask']
            temp.append(v)
        temp = sorted(temp, key=lambda k, sk='name': k[sk])
        objs = temp
        facts = {}
        if objs:
            facts['l3_interfaces'] = []
            params = utils.validate_config(self.argument_spec, {'config': objs})
            for cfg in params['config']:
                facts['l3_interfaces'].append(utils.remove_empties(cfg))
            facts['l3_interfaces'] = sorted(facts['l3_interfaces'], key=lambda k, sk='name': k[sk])
        ansible_facts['ansible_network_resources'].update(facts)
        return ansible_facts