from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from copy import deepcopy
from ansible.module_utils.common.dict_transformations import (
class AzureRMApplicationGateways(AzureRMModuleBase):
    """Configuration class for an Azure RM Application Gateway resource"""

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str'), sku=dict(type='dict', options=sku_spec), ssl_policy=dict(type='dict', options=ssl_policy_spec), gateway_ip_configurations=dict(type='list', elements='dict', options=dict(name=dict(type='str'), subnet=dict(type='dict', options=dict(id=dict(type='str'), name=dict(type='str'), virtual_network_name=dict(type='str'))))), authentication_certificates=dict(type='list', elements='dict', options=dict(name=dict(type='str'), data=dict(type='str'))), ssl_certificates=dict(type='list', elements='dict', options=dict(data=dict(type='str'), password=dict(type='str', no_log=True), name=dict(type='str'))), trusted_root_certificates=dict(type='list', elements='dict', options=trusted_root_certificates_spec), redirect_configurations=dict(type='list', elements='dict', options=redirect_configuration_spec), rewrite_rule_sets=dict(type='list', elements='dict', options=rewrite_rule_set_spec), frontend_ip_configurations=dict(type='list', elements='dict', options=dict(private_ip_address=dict(type='str'), private_ip_allocation_method=dict(type='str', choices=['static', 'dynamic']), public_ip_address=dict(type='raw'), name=dict(type='str'), subnet=dict(type='dict', options=dict(id=dict(type='str'), name=dict(type='str'), virtual_network_name=dict(type='str'))))), frontend_ports=dict(type='list', elements='dict', options=dict(port=dict(type='str'), name=dict(type='str'))), backend_address_pools=dict(type='list', elements='dict', options=dict(name=dict(type='str'), backend_addresses=dict(type='list', elements='dict', options=dict(fqdn=dict(type='str'), ip_address=dict(type='str'))))), backend_http_settings_collection=dict(type='list', elements='dict', options=dict(probe=dict(type='raw'), port=dict(type='int'), protocol=dict(type='str', choices=['http', 'https']), cookie_based_affinity=dict(type='str', choices=['enabled', 'disabled']), connection_draining=dict(type='dict', options=dict(drain_timeout_in_sec=dict(type='int'), enabled=dict(type='bool'))), request_timeout=dict(type='int'), authentication_certificates=dict(type='list', elements='dict', options=dict(id=dict(type='str'))), trusted_root_certificates=dict(type='list', elements='raw'), host_name=dict(type='str'), pick_host_name_from_backend_address=dict(type='bool'), affinity_cookie_name=dict(type='str'), path=dict(type='str'), name=dict(type='str'))), probes=dict(type='list', elements='dict', options=probe_spec), http_listeners=dict(type='list', elements='dict', options=dict(frontend_ip_configuration=dict(type='raw'), frontend_port=dict(type='raw'), protocol=dict(type='str', choices=['http', 'https']), host_name=dict(type='str'), ssl_certificate=dict(type='raw'), require_server_name_indication=dict(type='bool'), name=dict(type='str'))), url_path_maps=dict(type='list', elements='dict', options=url_path_maps_spec, mutually_exclusive=[('default_backend_address_pool', 'default_redirect_configuration')], required_one_of=[('default_backend_address_pool', 'default_redirect_configuration')], required_together=[('default_backend_address_pool', 'default_backend_http_settings')]), request_routing_rules=dict(type='list', elements='dict', options=dict(rule_type=dict(type='str', choices=['basic', 'path_based_routing']), backend_address_pool=dict(type='raw'), backend_http_settings=dict(type='raw'), http_listener=dict(type='raw'), name=dict(type='str'), redirect_configuration=dict(type='raw'), rewrite_rule_set=dict(type='raw'), url_path_map=dict(type='raw'))), autoscale_configuration=dict(type='dict', options=autoscale_configuration_spec), web_application_firewall_configuration=dict(type='dict', options=web_application_firewall_configuration_spec), enable_http2=dict(type='bool', default=False), gateway_state=dict(type='str', choices=['started', 'stopped']), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.name = None
        self.parameters = dict()
        self.results = dict(changed=False)
        self.state = None
        self.gateway_state = None
        self.to_do = Actions.NoAction
        super(AzureRMApplicationGateways, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                if key == 'id':
                    self.parameters['id'] = kwargs[key]
                elif key == 'location':
                    self.parameters['location'] = kwargs[key]
                elif key == 'sku':
                    ev = kwargs[key]
                    if 'name' in ev:
                        if ev['name'] == 'standard_small':
                            ev['name'] = 'Standard_Small'
                        elif ev['name'] == 'standard_medium':
                            ev['name'] = 'Standard_Medium'
                        elif ev['name'] == 'standard_large':
                            ev['name'] = 'Standard_Large'
                        elif ev['name'] == 'standard_v2':
                            ev['name'] = 'Standard_v2'
                        elif ev['name'] == 'waf_medium':
                            ev['name'] = 'WAF_Medium'
                        elif ev['name'] == 'waf_large':
                            ev['name'] = 'WAF_Large'
                        elif ev['name'] == 'waf_v2':
                            ev['name'] = 'WAF_v2'
                    if 'tier' in ev:
                        if ev['tier'] == 'standard':
                            ev['tier'] = 'Standard'
                        if ev['tier'] == 'standard_v2':
                            ev['tier'] = 'Standard_v2'
                        elif ev['tier'] == 'waf':
                            ev['tier'] = 'WAF'
                        elif ev['tier'] == 'waf_v2':
                            ev['tier'] = 'WAF_v2'
                    self.parameters['sku'] = ev
                elif key == 'ssl_policy':
                    ev = kwargs[key]
                    if 'policy_type' in ev:
                        ev['policy_type'] = _snake_to_camel(ev['policy_type'], True)
                    if 'policy_name' in ev:
                        if ev['policy_name'] == 'ssl_policy20150501':
                            ev['policy_name'] = 'AppGwSslPolicy20150501'
                        elif ev['policy_name'] == 'ssl_policy20170401':
                            ev['policy_name'] = 'AppGwSslPolicy20170401'
                        elif ev['policy_name'] == 'ssl_policy20170401_s':
                            ev['policy_name'] = 'AppGwSslPolicy20170401S'
                    if 'min_protocol_version' in ev:
                        if ev['min_protocol_version'] == 'tls_v1_0':
                            ev['min_protocol_version'] = 'TLSv1_0'
                        elif ev['min_protocol_version'] == 'tls_v1_1':
                            ev['min_protocol_version'] = 'TLSv1_1'
                        elif ev['min_protocol_version'] == 'tls_v1_2':
                            ev['min_protocol_version'] = 'TLSv1_2'
                    if 'disabled_ssl_protocols' in ev:
                        protocols = ev['disabled_ssl_protocols']
                        if protocols is not None:
                            for i in range(len(protocols)):
                                if protocols[i] == 'tls_v1_0':
                                    protocols[i] = 'TLSv1_0'
                                elif protocols[i] == 'tls_v1_1':
                                    protocols[i] = 'TLSv1_1'
                                elif protocols[i] == 'tls_v1_2':
                                    protocols[i] = 'TLSv1_2'
                    if 'cipher_suites' in ev:
                        suites = ev['cipher_suites']
                        if suites is not None:
                            for i in range(len(suites)):
                                suites[i] = suites[i].upper()
                    for prop_name in ['policy_name', 'min_protocol_version', 'disabled_ssl_protocols', 'cipher_suites']:
                        if prop_name in ev and ev[prop_name] is None:
                            del ev[prop_name]
                    self.parameters['ssl_policy'] = ev
                elif key == 'gateway_ip_configurations':
                    ev = kwargs[key]
                    for i in range(len(ev)):
                        item = ev[i]
                        if 'subnet' in item and item['subnet'] is not None and ('name' in item['subnet']) and (item['subnet']['name'] is not None) and ('virtual_network_name' in item['subnet']) and (item['subnet']['virtual_network_name'] is not None):
                            id = subnet_id(self.subscription_id, kwargs['resource_group'], item['subnet']['virtual_network_name'], item['subnet']['name'])
                            item['subnet'] = {'id': id}
                    self.parameters['gateway_ip_configurations'] = kwargs[key]
                elif key == 'authentication_certificates':
                    self.parameters['authentication_certificates'] = kwargs[key]
                elif key == 'ssl_certificates':
                    self.parameters['ssl_certificates'] = kwargs[key]
                elif key == 'trusted_root_certificates':
                    self.parameters['trusted_root_certificates'] = kwargs[key]
                elif key == 'redirect_configurations':
                    ev = kwargs[key]
                    for i in range(len(ev)):
                        item = ev[i]
                        if 'redirect_type' in item:
                            item['redirect_type'] = _snake_to_camel(item['redirect_type'], True)
                        if 'target_listener' in item:
                            id = http_listener_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], item['target_listener'])
                            item['target_listener'] = {'id': id}
                        if item['request_routing_rules']:
                            for j in range(len(item['request_routing_rules'])):
                                rule_name = item['request_routing_rules'][j]
                                id = request_routing_rule_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], rule_name)
                                item['request_routing_rules'][j] = {'id': id}
                        else:
                            del item['request_routing_rules']
                        if item['url_path_maps']:
                            for j in range(len(item['url_path_maps'])):
                                pathmap_name = item['url_path_maps'][j]
                                id = url_path_map_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], pathmap_name)
                                item['url_path_maps'][j] = {'id': id}
                        else:
                            del item['url_path_maps']
                        if item['path_rules']:
                            for j in range(len(item['path_rules'])):
                                pathrule = item['path_rules'][j]
                                if 'name' in pathrule and 'path_map_name' in pathrule:
                                    id = url_path_rule_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], pathrule['path_map_name'], pathrule['name'])
                                    item['path_rules'][j] = {'id': id}
                        else:
                            del item['path_rules']
                    self.parameters['redirect_configurations'] = ev
                elif key == 'rewrite_rule_sets':
                    ev = kwargs[key]
                    for i in range(len(ev)):
                        ev2 = ev[i]['rewrite_rules']
                        for j in range(len(ev2)):
                            item2 = ev2[j]
                            if item2['action_set'].get('url_configuration'):
                                if not item2['action_set']['url_configuration'].get('modified_path'):
                                    del item2['action_set']['url_configuration']['modified_path']
                                if not item2['action_set']['url_configuration'].get('modified_query_string'):
                                    del item2['action_set']['url_configuration']['modified_query_string']
                            else:
                                del item2['action_set']['url_configuration']
                    self.parameters['rewrite_rule_sets'] = ev
                elif key == 'frontend_ip_configurations':
                    ev = kwargs[key]
                    for i in range(len(ev)):
                        item = ev[i]
                        if 'private_ip_allocation_method' in item and item['private_ip_allocation_method'] is not None:
                            item['private_ip_allocation_method'] = _snake_to_camel(item['private_ip_allocation_method'], True)
                        if 'public_ip_address' in item and item['public_ip_address'] is not None:
                            id = public_ip_id(self.subscription_id, kwargs['resource_group'], item['public_ip_address'])
                            item['public_ip_address'] = {'id': id}
                        if 'subnet' in item and item['subnet'] is not None and ('name' in item['subnet']) and (item['subnet']['name'] is not None) and ('virtual_network_name' in item['subnet']) and (item['subnet']['virtual_network_name'] is not None):
                            id = subnet_id(self.subscription_id, kwargs['resource_group'], item['subnet']['virtual_network_name'], item['subnet']['name'])
                            item['subnet'] = {'id': id}
                    self.parameters['frontend_ip_configurations'] = ev
                elif key == 'frontend_ports':
                    self.parameters['frontend_ports'] = kwargs[key]
                elif key == 'backend_address_pools':
                    self.parameters['backend_address_pools'] = kwargs[key]
                elif key == 'probes':
                    ev = kwargs[key]
                    for i in range(len(ev)):
                        item = ev[i]
                        if 'protocol' in item and item['protocol'] is not None:
                            item['protocol'] = _snake_to_camel(item['protocol'], True)
                        if 'pick_host_name_from_backend_http_settings' in item and item['pick_host_name_from_backend_http_settings'] and ('host' in item):
                            del item['host']
                    self.parameters['probes'] = ev
                elif key == 'backend_http_settings_collection':
                    ev = kwargs[key]
                    for i in range(len(ev)):
                        item = ev[i]
                        if 'protocol' in item and item['protocol'] is not None:
                            item['protocol'] = _snake_to_camel(item['protocol'], True)
                        if 'cookie_based_affinity' in item and item['cookie_based_affinity'] is not None:
                            item['cookie_based_affinity'] = _snake_to_camel(item['cookie_based_affinity'], True)
                        if 'probe' in item and item['probe'] is not None:
                            id = probe_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], item['probe'])
                            item['probe'] = {'id': id}
                        if 'trusted_root_certificates' in item and item['trusted_root_certificates'] is not None:
                            for j in range(len(item['trusted_root_certificates'])):
                                id = item['trusted_root_certificates'][j]
                                id = id if is_valid_resource_id(id) else trusted_root_certificate_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], id)
                                item['trusted_root_certificates'][j] = {'id': id}
                    self.parameters['backend_http_settings_collection'] = ev
                elif key == 'http_listeners':
                    ev = kwargs[key]
                    for i in range(len(ev)):
                        item = ev[i]
                        if 'frontend_ip_configuration' in item and item['frontend_ip_configuration'] is not None:
                            id = frontend_ip_configuration_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], item['frontend_ip_configuration'])
                            item['frontend_ip_configuration'] = {'id': id}
                        if 'frontend_port' in item and item['frontend_port'] is not None:
                            id = frontend_port_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], item['frontend_port'])
                            item['frontend_port'] = {'id': id}
                        if 'ssl_certificate' in item and item['ssl_certificate'] is not None:
                            id = ssl_certificate_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], item['ssl_certificate'])
                            item['ssl_certificate'] = {'id': id}
                        if 'protocol' in item and item['protocol'] is not None:
                            item['protocol'] = _snake_to_camel(item['protocol'], True)
                        ev[i] = item
                    self.parameters['http_listeners'] = ev
                elif key == 'url_path_maps':
                    ev = kwargs[key]
                    for i in range(len(ev)):
                        item = ev[i]
                        if item['default_backend_address_pool'] and item['default_backend_address_pool'] is not None:
                            id = backend_address_pool_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], item['default_backend_address_pool'])
                            item['default_backend_address_pool'] = {'id': id}
                        else:
                            del item['default_backend_address_pool']
                        if item['default_backend_http_settings'] and item['default_backend_http_settings'] is not None:
                            id = backend_http_settings_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], item['default_backend_http_settings'])
                            item['default_backend_http_settings'] = {'id': id}
                        else:
                            del item['default_backend_http_settings']
                        if 'path_rules' in item:
                            ev2 = item['path_rules']
                            for j in range(len(ev2)):
                                item2 = ev2[j]
                                if item2['backend_address_pool'] and item2['backend_address_pool'] is not None:
                                    id = backend_address_pool_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], item2['backend_address_pool'])
                                    item2['backend_address_pool'] = {'id': id}
                                else:
                                    del item2['backend_address_pool']
                                if item2['backend_http_settings'] and item2['backend_http_settings'] is not None:
                                    id = backend_http_settings_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], item2['backend_http_settings'])
                                    item2['backend_http_settings'] = {'id': id}
                                else:
                                    del item2['backend_http_settings']
                                if item2['redirect_configuration'] and item2['redirect_configuration'] is not None:
                                    id = redirect_configuration_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], item2['redirect_configuration'])
                                    item2['redirect_configuration'] = {'id': id}
                                else:
                                    del item2['redirect_configuration']
                                if item2['rewrite_rule_set']:
                                    id = item2['rewrite_rule_set']
                                    id = id if is_valid_resource_id(id) else rewrite_rule_set_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], id)
                                    item2['rewrite_rule_set'] = {'id': id}
                                else:
                                    del item2['rewrite_rule_set']
                                ev2[j] = item2
                        if item['default_redirect_configuration']:
                            id = redirect_configuration_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], item['default_redirect_configuration'])
                            item['default_redirect_configuration'] = {'id': id}
                        else:
                            del item['default_redirect_configuration']
                        if item['default_rewrite_rule_set']:
                            id = item['default_rewrite_rule_set']
                            id = id if is_valid_resource_id(id) else rewrite_rule_set_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], id)
                            item['default_rewrite_rule_set'] = {'id': id}
                        else:
                            del item['default_rewrite_rule_set']
                        ev[i] = item
                    self.parameters['url_path_maps'] = ev
                elif key == 'request_routing_rules':
                    ev = kwargs[key]
                    for i in range(len(ev)):
                        item = ev[i]
                        if 'rule_type' in item and item['rule_type'] is not None and (item['rule_type'] == 'path_based_routing') and ('backend_address_pool' in item) and (item['backend_address_pool'] is not None):
                            del item['backend_address_pool']
                        if 'backend_address_pool' in item and item['backend_address_pool'] is not None:
                            id = backend_address_pool_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], item['backend_address_pool'])
                            item['backend_address_pool'] = {'id': id}
                        if 'backend_http_settings' in item and item['backend_http_settings'] is not None:
                            id = backend_http_settings_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], item['backend_http_settings'])
                            item['backend_http_settings'] = {'id': id}
                        if 'http_listener' in item and item['http_listener'] is not None:
                            id = http_listener_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], item['http_listener'])
                            item['http_listener'] = {'id': id}
                        if 'protocol' in item and item['protocol'] is not None:
                            item['protocol'] = _snake_to_camel(item['protocol'], True)
                        if 'rule_type' in item and item['rule_type'] is not None:
                            item['rule_type'] = _snake_to_camel(item['rule_type'], True)
                        if 'redirect_configuration' in item and item['redirect_configuration'] is not None:
                            id = redirect_configuration_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], item['redirect_configuration'])
                            item['redirect_configuration'] = {'id': id}
                        if 'url_path_map' in item and item['url_path_map'] is not None:
                            id = url_path_map_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], item['url_path_map'])
                            item['url_path_map'] = {'id': id}
                        if item.get('rewrite_rule_set'):
                            id = item.get('rewrite_rule_set')
                            id = id if is_valid_resource_id(id) else rewrite_rule_set_id(self.subscription_id, kwargs['resource_group'], kwargs['name'], id)
                            item['rewrite_rule_set'] = {'id': id}
                        ev[i] = item
                    self.parameters['request_routing_rules'] = ev
                elif key == 'etag':
                    self.parameters['etag'] = kwargs[key]
                elif key == 'autoscale_configuration':
                    self.parameters['autoscale_configuration'] = kwargs[key]
                elif key == 'web_application_firewall_configuration':
                    self.parameters['web_application_firewall_configuration'] = kwargs[key]
                elif key == 'enable_http2':
                    self.parameters['enable_http2'] = kwargs[key]
        response = None
        resource_group = self.get_resource_group(self.resource_group)
        if 'location' not in self.parameters:
            self.parameters['location'] = resource_group.location
        old_response = self.get_applicationgateway()
        if not old_response:
            self.log("Application Gateway instance doesn't exist")
            if self.state == 'absent':
                self.log("Old instance didn't exist")
            else:
                self.to_do = Actions.Create
        else:
            self.log('Application Gateway instance already exists')
            if self.state == 'absent':
                self.to_do = Actions.Delete
            elif self.state == 'present':
                self.log('Need to check if Application Gateway instance has to be deleted or may be updated')
                self.to_do = Actions.Update
        if self.to_do == Actions.Update:
            if old_response['operational_state'] == 'Stopped' and self.gateway_state == 'started':
                self.to_do = Actions.Start
            elif old_response['operational_state'] == 'Running' and self.gateway_state == 'stopped':
                self.to_do = Actions.Stop
            elif old_response['operational_state'] == 'Stopped' and self.gateway_state == 'stopped' or (old_response['operational_state'] == 'Running' and self.gateway_state == 'started'):
                self.to_do = Actions.NoAction
            elif self.parameters['location'] != old_response['location'] or self.parameters['enable_http2'] != old_response['enable_http2'] or self.parameters['sku']['name'] != old_response['sku']['name'] or (self.parameters['sku']['tier'] != old_response['sku']['tier']) or (self.parameters['sku'].get('capacity', None) != old_response['sku'].get('capacity', None)) or (not compare_arrays(old_response, self.parameters, 'authentication_certificates')) or (not compare_dicts(old_response, self.parameters, 'ssl_policy')) or (not compare_arrays(old_response, self.parameters, 'gateway_ip_configurations')) or (not compare_arrays(old_response, self.parameters, 'redirect_configurations')) or (not compare_arrays(old_response, self.parameters, 'rewrite_rule_sets')) or (not compare_arrays(old_response, self.parameters, 'frontend_ip_configurations')) or (not compare_arrays(old_response, self.parameters, 'frontend_ports')) or (not compare_arrays(old_response, self.parameters, 'backend_address_pools')) or (not compare_arrays(old_response, self.parameters, 'probes')) or (not compare_arrays(old_response, self.parameters, 'backend_http_settings_collection')) or (not compare_arrays(old_response, self.parameters, 'request_routing_rules')) or (not compare_arrays(old_response, self.parameters, 'http_listeners')) or (not compare_arrays(old_response, self.parameters, 'url_path_maps')) or (not compare_arrays(old_response, self.parameters, 'trusted_root_certificates')) or (not compare_dicts(old_response, self.parameters, 'autoscale_configuration')) or (not compare_dicts(old_response, self.parameters, 'web_application_firewall_configuration')):
                self.to_do = Actions.Update
            else:
                self.to_do = Actions.NoAction
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.log('Need to Create / Update the Application Gateway instance')
            if self.check_mode:
                self.results['changed'] = True
                self.results['parameters'] = self.parameters
                return self.results
            response = self.create_update_applicationgateway()
            if not old_response:
                self.results['changed'] = True
            else:
                self.results['changed'] = old_response.__ne__(response)
            self.log('Creation / Update done')
        elif self.to_do == Actions.Start or self.to_do == Actions.Stop:
            self.log('Need to Start / Stop the Application Gateway instance')
            self.results['changed'] = True
            response = old_response
            if self.check_mode:
                return self.results
            elif self.to_do == Actions.Start:
                self.start_applicationgateway()
                response['operational_state'] = 'Running'
            else:
                self.stop_applicationgateway()
                response['operational_state'] = 'Stopped'
        elif self.to_do == Actions.Delete:
            self.log('Application Gateway instance deleted')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_applicationgateway()
            while self.get_applicationgateway():
                time.sleep(20)
        else:
            self.log('Application Gateway instance unchanged')
            self.results['changed'] = False
            response = old_response
        if response:
            self.results.update(self.format_response(response))
        return self.results

    def create_update_applicationgateway(self):
        """
        Creates or updates Application Gateway with the specified configuration.

        :return: deserialized Application Gateway instance state dictionary
        """
        self.log('Creating / Updating the Application Gateway instance {0}'.format(self.name))
        try:
            response = self.network_client.application_gateways.begin_create_or_update(resource_group_name=self.resource_group, application_gateway_name=self.name, parameters=self.parameters)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.log('Error attempting to create the Application Gateway instance.')
            self.fail('Error creating the Application Gateway instance: {0}'.format(str(exc)))
        return response.as_dict()

    def delete_applicationgateway(self):
        """
        Deletes specified Application Gateway instance in the specified subscription and resource group.

        :return: True
        """
        self.log('Deleting the Application Gateway instance {0}'.format(self.name))
        try:
            response = self.network_client.application_gateways.begin_delete(resource_group_name=self.resource_group, application_gateway_name=self.name)
        except Exception as e:
            self.log('Error attempting to delete the Application Gateway instance.')
            self.fail('Error deleting the Application Gateway instance: {0}'.format(str(e)))
        return True

    def get_applicationgateway(self):
        """
        Gets the properties of the specified Application Gateway.

        :return: deserialized Application Gateway instance state dictionary
        """
        self.log('Checking if the Application Gateway instance {0} is present'.format(self.name))
        found = False
        try:
            response = self.network_client.application_gateways.get(resource_group_name=self.resource_group, application_gateway_name=self.name)
            found = True
            self.log('Response : {0}'.format(response))
            self.log('Application Gateway instance : {0} found'.format(response.name))
        except ResourceNotFoundError as e:
            self.log('Did not find the Application Gateway instance.')
        if found is True:
            return response.as_dict()
        return False

    def start_applicationgateway(self):
        self.log('Starting the Application Gateway instance {0}'.format(self.name))
        try:
            response = self.network_client.application_gateways.begin_start(resource_group_name=self.resource_group, application_gateway_name=self.name)
            if isinstance(response, LROPoller):
                self.get_poller_result(response)
        except Exception as e:
            self.log('Error attempting to start the Application Gateway instance.')
            self.fail('Error starting the Application Gateway instance: {0}'.format(str(e)))

    def stop_applicationgateway(self):
        self.log('Stopping the Application Gateway instance {0}'.format(self.name))
        try:
            response = self.network_client.application_gateways.begin_stop(resource_group_name=self.resource_group, application_gateway_name=self.name)
            if isinstance(response, LROPoller):
                self.get_poller_result(response)
        except Exception as e:
            self.log('Error attempting to stop the Application Gateway instance.')
            self.fail('Error stopping the Application Gateway instance: {0}'.format(str(e)))

    def format_response(self, appgw_dict):
        id = appgw_dict.get('id')
        id_dict = parse_resource_id(id)
        d = {'id': id, 'name': appgw_dict.get('name'), 'resource_group': id_dict.get('resource_group', self.resource_group), 'location': appgw_dict.get('location'), 'operational_state': appgw_dict.get('operational_state'), 'provisioning_state': appgw_dict.get('provisioning_state')}
        return d