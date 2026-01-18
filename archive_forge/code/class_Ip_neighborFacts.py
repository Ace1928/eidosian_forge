from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.ip_neighbor.ip_neighbor import Ip_neighborArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
class Ip_neighborFacts(object):
    """ The sonic ip_neighbor fact class
    """

    def __init__(self, module, subspec='config', options='options'):
        self._module = module
        self.argument_spec = Ip_neighborArgs.argument_spec
        spec = deepcopy(self.argument_spec)
        if subspec:
            if options:
                facts_argument_spec = spec[subspec][options]
            else:
                facts_argument_spec = spec[subspec]
        else:
            facts_argument_spec = spec
        self.generated_spec = utils.generate_dict(facts_argument_spec)

    def populate_facts(self, connection, ansible_facts, data=None):
        """ Populate the facts for ip_neighbor
        :param connection: the device connection
        :param ansible_facts: Facts dictionary
        :param data: previously collected conf
        :rtype: dictionary
        :returns: facts
        """
        if not data:
            data = self.get_ip_neighbor_global()
        objs = self.render_config(self.generated_spec, data)
        ansible_facts['ansible_network_resources'].pop('ip_neighbor', None)
        facts = {}
        if objs:
            params = utils.validate_config(self.argument_spec, {'config': objs})
            facts['ip_neighbor'] = params['config']
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
        return conf

    def get_ip_neighbor_global(self):
        """Get IP neighbor global configurations"""
        config_path = 'data/openconfig-neighbor:neighbor-globals/neighbor-global=Values/config'
        config_request = [{'path': config_path, 'method': GET}]
        config_response = []
        ip_neigh_glb_conf = dict()
        try:
            config_response = edit_config(self._module, to_request(self._module, config_request))
        except ConnectionError as exc:
            if re.search('code.*404', str(exc)):
                return ip_neigh_glb_conf
            else:
                self._module.fail_json(msg=str(exc), code=exc.code)
        config = dict()
        if 'openconfig-neighbor:config' in config_response[0][1]:
            config = config_response[0][1].get('openconfig-neighbor:config', {})
        if 'ipv4-arp-timeout' in config:
            ip_neigh_glb_conf['ipv4_arp_timeout'] = config['ipv4-arp-timeout']
        if 'ipv4-drop-neighbor-aging-time' in config:
            ip_neigh_glb_conf['ipv4_drop_neighbor_aging_time'] = config['ipv4-drop-neighbor-aging-time']
        if 'ipv6-drop-neighbor-aging-time' in config:
            ip_neigh_glb_conf['ipv6_drop_neighbor_aging_time'] = config['ipv6-drop-neighbor-aging-time']
        if 'ipv6-nd-cache-expiry' in config:
            ip_neigh_glb_conf['ipv6_nd_cache_expiry'] = config['ipv6-nd-cache-expiry']
        if 'num-local-neigh' in config:
            ip_neigh_glb_conf['num_local_neigh'] = config['num-local-neigh']
        return ip_neigh_glb_conf