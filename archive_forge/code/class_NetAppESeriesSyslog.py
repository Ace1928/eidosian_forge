from __future__ import absolute_import, division, print_function
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
class NetAppESeriesSyslog(NetAppESeriesModule):

    def __init__(self):
        ansible_options = dict(state=dict(choices=['present', 'absent'], required=False, default='present'), address=dict(type='str', required=False), port=dict(type='int', default=514, required=False), protocol=dict(choices=['tcp', 'tls', 'udp'], default='udp', required=False), components=dict(type='list', required=False, default=['auditLog']), test=dict(type='bool', default=False, require=False))
        required_if = [['state', 'present', ['address', 'port', 'protocol', 'components']]]
        mutually_exclusive = [['test', 'absent']]
        super(NetAppESeriesSyslog, self).__init__(ansible_options=ansible_options, web_services_version='02.00.0000.0000', mutually_exclusive=mutually_exclusive, required_if=required_if, supports_check_mode=True)
        args = self.module.params
        self.syslog = args['state'] in ['present']
        self.address = args['address']
        self.port = args['port']
        self.protocol = args['protocol']
        self.components = args['components']
        self.test = args['test']
        self.ssid = args['ssid']
        self.url = args['api_url']
        self.creds = dict(url_password=args['api_password'], validate_certs=args['validate_certs'], url_username=args['api_username'])
        self.components.sort()
        self.check_mode = self.module.check_mode
        self.url_path_prefix = ''
        if not self.is_embedded() and self.ssid != '0' and (self.ssid.lower() != 'proxy'):
            self.url_path_prefix = 'storage-systems/%s/forward/devmgr/v2/' % self.ssid

    def get_configuration(self):
        """Retrieve existing syslog configuration."""
        try:
            rc, result = self.request(self.url_path_prefix + 'storage-systems/%s/syslog' % self.ssid)
            return result
        except Exception as err:
            self.module.fail_json(msg='Failed to retrieve syslog configuration! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))

    def test_configuration(self, body):
        """Send test syslog message to the storage array.

        Allows fix number of retries to occur before failure is issued to give the storage array time to create
        new syslog server record.
        """
        try:
            rc, result = self.request(self.url_path_prefix + 'storage-systems/%s/syslog/%s/test' % (self.ssid, body['id']), method='POST')
        except Exception as err:
            self.module.fail_json(msg='We failed to send test message! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))

    def update_configuration(self):
        """Post the syslog request to array."""
        config_match = None
        perfect_match = None
        update = False
        body = dict()
        configs = self.get_configuration()
        if self.address:
            for config in configs:
                if config['serverAddress'] == self.address:
                    config_match = config
                    if config['port'] == self.port and config['protocol'] == self.protocol and (len(config['components']) == len(self.components)) and all([component['type'] in self.components for component in config['components']]):
                        perfect_match = config_match
                        break
        if self.syslog:
            if not perfect_match:
                update = True
                if config_match:
                    body.update(dict(id=config_match['id']))
                components = [dict(type=component_type) for component_type in self.components]
                body.update(dict(serverAddress=self.address, port=self.port, protocol=self.protocol, components=components))
                self.make_configuration_request(body)
        elif config_match:
            if self.address:
                update = True
                body.update(dict(id=config_match['id']))
                self.make_configuration_request(body)
            elif configs:
                update = True
                for config in configs:
                    body.update(dict(id=config['id']))
                    self.make_configuration_request(body)
        return update

    def make_configuration_request(self, body):
        if not self.check_mode:
            try:
                if self.syslog:
                    if 'id' in body:
                        rc, result = self.request(self.url_path_prefix + 'storage-systems/%s/syslog/%s' % (self.ssid, body['id']), method='POST', data=body)
                    else:
                        rc, result = self.request(self.url_path_prefix + 'storage-systems/%s/syslog' % self.ssid, method='POST', data=body)
                        body.update(result)
                    if self.test:
                        self.test_configuration(body)
                elif 'id' in body:
                    rc, result = self.request(self.url_path_prefix + 'storage-systems/%s/syslog/%s' % (self.ssid, body['id']), method='DELETE')
            except Exception as err:
                self.module.fail_json(msg='We failed to modify syslog configuration! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))

    def update(self):
        """Update configuration and respond to ansible."""
        update = self.update_configuration()
        self.module.exit_json(msg='The syslog settings have been updated.', changed=update)