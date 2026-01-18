from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.vlans.vlans import (
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.utils.utils import (
class VlansFacts(object):
    """The junos vlans fact class"""

    def __init__(self, module, subspec='config', options='options'):
        self._module = module
        self.argument_spec = VlansArgs.argument_spec
        spec = deepcopy(self.argument_spec)
        if subspec:
            if options:
                facts_argument_spec = spec[subspec][options]
            else:
                facts_argument_spec = spec[subspec]
        else:
            facts_argument_spec = spec
        self.generated_spec = utils.generate_dict(facts_argument_spec)

    def get_device_data(self, connection, config_filter):
        """
        :param connection:
        :param config_filter:
        :return:
        """
        return get_resource_config(connection, config_filter=config_filter)

    def populate_facts(self, connection, ansible_facts, data=None):
        """Populate the facts for vlans
        :param connection: the device connection
        :param ansible_facts: Facts dictionary
        :param data: previously collected conf
        :rtype: dictionary
        :returns: facts
        """
        if not HAS_LXML:
            self._module.fail_json(msg='lxml is not installed.')
        if not data:
            config_filter = '\n                <configuration>\n                  <vlans>\n                  </vlans>\n                </configuration>\n                '
            data = self.get_device_data(connection, config_filter)
        if isinstance(data, string_types):
            data = etree.fromstring(to_bytes(data, errors='surrogate_then_replace'))
        resources = data.xpath('configuration/vlans/vlan')
        objs = []
        for resource in resources:
            if resource is not None:
                obj = self.render_config(self.generated_spec, resource)
                if obj:
                    objs.append(obj)
        facts = {}
        if objs:
            facts['vlans'] = []
            params = utils.validate_config(self.argument_spec, {'config': objs})
            for cfg in params['config']:
                facts['vlans'].append(utils.remove_empties(cfg))
        ansible_facts['ansible_network_resources'].update(facts)
        return ansible_facts

    def render_config(self, spec, conf):
        """
        Render config as dictionary structure and delete keys
          from spec for null values

        :param spec: The facts tree, generated from the argspec
        :param conf: The configuration
        :rtype: dictionary
        :returns: The generated config
        """
        config = deepcopy(spec)
        config['name'] = utils.get_xml_conf_arg(conf, 'name')
        config['vlan_id'] = utils.get_xml_conf_arg(conf, 'vlan-id')
        config['description'] = utils.get_xml_conf_arg(conf, 'description')
        config['l3_interface'] = utils.get_xml_conf_arg(conf, 'l3-interface')
        return utils.remove_empties(config)