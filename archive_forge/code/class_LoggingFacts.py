from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.logging.logging import LoggingArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
class LoggingFacts(object):
    """ The sonic logging fact class
    """

    def __init__(self, module, subspec='config', options='options'):
        self._module = module
        self.argument_spec = LoggingArgs.argument_spec
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
        """ Populate the facts for logging
        :param connection: the device connection
        :param ansible_facts: Facts dictionary
        :param data: previously collected conf
        :rtype: dictionary
        :returns: facts
        """
        if not data:
            data = self.get_logging_configuration()
        obj = self.render_config(self.generated_spec, data)
        ansible_facts['ansible_network_resources'].pop('logging', None)
        facts = {}
        if obj:
            params = utils.validate_config(self.argument_spec, {'config': obj})
            facts['logging'] = params['config']
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

    def get_logging_configuration(self):
        """Get all logging configuration"""
        config_request = [{'path': 'data/openconfig-system:system/logging', 'method': GET}]
        config_response = []
        try:
            config_response = edit_config(self._module, to_request(self._module, config_request))
        except ConnectionError as exc:
            self._module.fail_json(msg=str(exc), code=exc.code)
        logging_response = dict()
        if 'openconfig-system:logging' in config_response[0][1]:
            logging_response = config_response[0][1].get('openconfig-system:logging', {})
        remote_servers = []
        if 'remote-servers' in logging_response:
            remote_servers = logging_response['remote-servers'].get('remote-server', [])
        logging_config = dict()
        logging_servers = []
        for remote_server in remote_servers:
            rs_config = remote_server.get('config', {})
            logging_server = {}
            logging_server['host'] = rs_config['host']
            if 'openconfig-system-ext:message-type' in rs_config:
                logging_server['message_type'] = rs_config['openconfig-system-ext:message-type']
            if 'openconfig-system-ext:source-interface' in rs_config:
                logging_server['source_interface'] = rs_config['openconfig-system-ext:source-interface']
                if logging_server['source_interface'].startswith('Management') or logging_server['source_interface'].startswith('Mgmt'):
                    logging_server['source_interface'] = 'eth0'
            if 'openconfig-system-ext:vrf-name' in rs_config:
                logging_server['vrf'] = rs_config['openconfig-system-ext:vrf-name']
            if 'remote-port' in rs_config:
                logging_server['remote_port'] = rs_config['remote-port']
            logging_servers.append(logging_server)
        logging_config['remote_servers'] = logging_servers
        return logging_config