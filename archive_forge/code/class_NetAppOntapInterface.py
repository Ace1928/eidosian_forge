from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
class NetAppOntapInterface:
    """ object to describe  interface info """

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, choices=['present', 'absent'], default='present'), interface_name=dict(required=True, type='str'), interface_type=dict(type='str', choices=['fc', 'ip']), ipspace=dict(type='str'), broadcast_domain=dict(type='str'), home_node=dict(required=False, type='str', default=None), current_node=dict(required=False, type='str'), home_port=dict(required=False, type='str'), current_port=dict(required=False, type='str'), role=dict(required=False, type='str'), is_ipv4_link_local=dict(required=False, type='bool', default=None), address=dict(required=False, type='str'), netmask=dict(required=False, type='str'), vserver=dict(required=False, type='str'), firewall_policy=dict(required=False, type='str', default=None), failover_policy=dict(required=False, type='str', default=None, choices=['disabled', 'system-defined', 'local-only', 'sfo-partner-only', 'broadcast-domain-wide']), failover_scope=dict(required=False, type='str', default=None, choices=['home_port_only', 'default', 'home_node_only', 'sfo_partners_only', 'broadcast_domain_only']), failover_group=dict(required=False, type='str'), admin_status=dict(required=False, choices=['up', 'down']), subnet_name=dict(required=False, type='str'), is_auto_revert=dict(required=False, type='bool', default=None), protocols=dict(required=False, type='list', elements='str'), data_protocol=dict(required=False, type='str', choices=['fc_nvme', 'fcp']), force_subnet_association=dict(required=False, type='bool', default=None), dns_domain_name=dict(required=False, type='str'), listen_for_dns_query=dict(required=False, type='bool'), is_dns_update_enabled=dict(required=False, type='bool'), service_policy=dict(required=False, type='str', default=None), from_name=dict(required=False, type='str'), ignore_zapi_options=dict(required=False, type='list', elements='str', default=['force_subnet_association'], choices=REST_IGNORABLE_OPTIONS), probe_port=dict(required=False, type='int'), fail_if_subnet_conflicts=dict(required=False, type='bool')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, mutually_exclusive=[['subnet_name', 'address'], ['subnet_name', 'netmask'], ['is_ipv4_link_local', 'address'], ['is_ipv4_link_local', 'netmask'], ['is_ipv4_link_local', 'subnet_name'], ['failover_policy', 'failover_scope']], supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = OntapRestAPI(self.module)
        unsupported_rest_properties = [key for key in REST_IGNORABLE_OPTIONS if key not in self.parameters['ignore_zapi_options']]
        unsupported_rest_properties.extend(REST_UNSUPPORTED_OPTIONS)
        if self.na_helper.safe_get(self.parameters, ['address']):
            self.parameters['address'] = netapp_ipaddress.validate_and_compress_ip_address(self.parameters['address'], self.module)
        partially_supported_rest_properties = [['dns_domain_name', (9, 9, 0)], ['is_dns_update_enabled', (9, 9, 1)], ['probe_port', (9, 10, 1)], ['subnet_name', (9, 11, 1)], ['fail_if_subnet_conflicts', (9, 11, 1)]]
        self.use_rest = self.rest_api.is_rest_supported_properties(self.parameters, unsupported_rest_properties, partially_supported_rest_properties)
        if self.use_rest and (not self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 7, 0)):
            msg = 'REST requires ONTAP 9.7 or later for interface APIs.'
            self.use_rest = self.na_helper.fall_back_to_zapi(self.module, msg, self.parameters)
        if self.use_rest:
            self.cluster_nodes = None
            self.home_node = None
            self.map_failover_policy()
            self.validate_rest_input_parameters()
            if self.parameters.get('netmask'):
                self.parameters['netmask'] = str(netapp_ipaddress.netmask_to_netmask_length(self.parameters.get('address'), self.parameters['netmask'], self.module))
        elif netapp_utils.has_netapp_lib() is False:
            self.module.fail_json(msg=netapp_utils.netapp_lib_is_required())
        else:
            for option in ('probe_port', 'fail_if_subnet_conflicts'):
                if self.parameters.get(option) is not None:
                    self.module.fail_json(msg='Error option %s requires REST.' % option)
            if 'vserver' not in self.parameters:
                self.module.fail_json(msg='missing required argument with ZAPI: vserver')
            self.server = netapp_utils.setup_na_ontap_zapi(module=self.module)
            if self.parameters.get('netmask'):
                self.parameters['netmask'] = netapp_ipaddress.netmask_length_to_netmask(self.parameters.get('address'), self.parameters['netmask'], self.module)

    def map_failover_policy(self):
        if self.use_rest and 'failover_policy' in self.parameters:
            mapping = dict(zip(FAILOVER_POLICIES, FAILOVER_SCOPES))
            self.parameters['failover_scope'] = mapping[self.parameters['failover_policy']]

    def set_interface_type(self, interface_type):
        if 'interface_type' in self.parameters:
            if self.parameters['interface_type'] != interface_type:
                self.module.fail_json(msg='Error: mismatch between configured interface_type: %s and derived interface_type: %s.' % (self.parameters['interface_type'], interface_type))
        else:
            self.parameters['interface_type'] = interface_type

    def derive_fc_data_protocol(self):
        protocols = self.parameters.get('protocols')
        if not protocols:
            return
        if len(protocols) > 1:
            self.module.fail_json(msg='A single protocol entry is expected for FC interface, got %s.' % protocols)
        mapping = {'fc-nvme': 'fc_nvme', 'fc_nvme': 'fc_nvme', 'fcp': 'fcp'}
        if protocols[0] not in mapping:
            self.module.fail_json(msg='Unexpected protocol value %s.' % protocols[0])
        data_protocol = mapping[protocols[0]]
        if 'data_protocol' in self.parameters and self.parameters['data_protocol'] != data_protocol:
            self.module.fail_json(msg='Error: mismatch between configured data_protocol: %s and data_protocols: %s' % (self.parameters['data_protocol'], protocols))
        self.parameters['data_protocol'] = data_protocol

    def derive_interface_type(self):
        protocols = self.parameters.get('protocols')
        if protocols in (None, ['none']):
            if self.parameters.get('role') in ('cluster', 'intercluster') or any((x in self.parameters for x in ('address', 'netmask', 'subnet_name'))):
                self.set_interface_type('ip')
            return
        protocol_types = set()
        unknown_protocols = []
        for protocol in protocols:
            if protocol.lower() in ['fc-nvme', 'fcp']:
                protocol_types.add('fc')
            elif protocol.lower() in ['nfs', 'cifs', 'iscsi']:
                protocol_types.add('ip')
            elif protocol.lower() != 'none':
                unknown_protocols.append(protocol)
        errors = []
        if unknown_protocols:
            errors.append('unexpected value(s) for protocols: %s' % unknown_protocols)
        if len(protocol_types) > 1:
            errors.append('incompatible value(s) for protocols: %s' % protocols)
        if errors:
            self.module.fail_json(msg='Error: unable to determine interface type, please set interface_type: %s' % ' - '.join(errors))
        if protocol_types:
            self.set_interface_type(protocol_types.pop())
        return

    def derive_block_file_type(self, protocols):
        block_p, file_p, fcp = (False, False, False)
        if protocols is None:
            fcp = self.parameters.get('interface_type') == 'fc'
            return (fcp, file_p, fcp)
        block_values, file_values = ([], [])
        for protocol in protocols:
            if protocol.lower() in ['fc-nvme', 'fcp', 'iscsi']:
                block_p = True
                block_values.append(protocol)
                if protocol.lower() in ['fc-nvme', 'fcp']:
                    fcp = True
            elif protocol.lower() in ['nfs', 'cifs']:
                file_p = True
                file_values.append(protocol)
        if block_p and file_p:
            self.module.fail_json(msg='Cannot use any of %s with %s' % (block_values, file_values))
        return (block_p, file_p, fcp)

    def get_interface_record_rest(self, if_type, query, fields):
        if 'ipspace' in self.parameters and if_type == 'ip':
            query['ipspace.name'] = self.parameters['ipspace']
        return rest_generic.get_one_record(self.rest_api, self.get_net_int_api(if_type), query, fields)

    def get_interface_records_rest(self, if_type, query, fields):
        if 'ipspace' in self.parameters:
            if if_type == 'ip':
                query['ipspace.name'] = self.parameters['ipspace']
            else:
                self.module.warn('ipspace is ignored for FC interfaces.')
        records, error = rest_generic.get_0_or_more_records(self.rest_api, self.get_net_int_api(if_type), query, fields)
        if error and 'are available in precluster.' in error:
            self.module.fail_json(msg='This module cannot use REST in precluster mode, ZAPI can be forced with use_rest: never.  Error: %s' % error)
        return (records, error)

    def get_net_int_api(self, if_type=None):
        if if_type is None:
            if_type = self.parameters.get('interface_type')
        if if_type is None:
            self.module.fail_json(msg='Error: missing option "interface_type (or could not be derived)')
        return 'network/%s/interfaces' % if_type

    def find_interface_record(self, records, home_node, name):
        full_name = '%s_%s' % (home_node, name) if home_node is not None else name
        full_name_records = [record for record in records if record['name'] == full_name]
        if len(full_name_records) > 1:
            self.module.fail_json(msg='Error: multiple records for: %s - %s' % (full_name, full_name_records))
        return full_name_records[0] if full_name_records else None

    def find_exact_match(self, records, name):
        """ with vserver, we expect an exact match
            but ONTAP transforms cluster interface names by prepending the home_port
        """
        if 'vserver' in self.parameters:
            if len(records) > 1:
                self.module.fail_json(msg='Error: unexpected records for name: %s, vserver: %s - %s' % (name, self.parameters['vserver'], records))
            return records[0] if records else None
        record = self.find_interface_record(records, None, name)
        if 'home_node' in self.parameters and self.parameters['home_node'] != 'localhost':
            home_record = self.find_interface_record(records, self.parameters['home_node'], name)
            if record and home_record:
                self.module.warn('Found both %s, selecting %s' % ([record['name'] for record in (record, home_record)], home_record['name']))
        else:
            home_node_records = []
            for home_node in self.get_cluster_node_names_rest():
                home_record = self.find_interface_record(records, home_node, name)
                if home_record:
                    home_node_records.append(home_record)
            if len(home_node_records) > 1:
                self.module.fail_json(msg='Error: multiple matches for name: %s: %s.  Set home_node parameter.' % (name, [record['name'] for record in home_node_records]))
            home_record = home_node_records[0] if home_node_records else None
            if record and home_node_records:
                self.module.fail_json(msg='Error: multiple matches for name: %s: %s.  Set home_node parameter.' % (name, [record['name'] for record in (record, home_record)]))
        if home_record:
            record = home_record
        if record and name == self.parameters['interface_name'] and (name != record['name']):
            self.parameters['interface_name'] = record['name']
            self.module.warn('adjusting name from %s to %s' % (name, record['name']))
        return record

    def get_interface_rest(self, name):
        """
        Return details about the interface
        :param:
            name : Name of the interface

        :return: Details about the interface. None if not found.
        :rtype: dict
        """
        self.derive_interface_type()
        if_type = self.parameters.get('interface_type')
        if 'vserver' in self.parameters:
            query_ip = {'name': name, 'svm.name': self.parameters['vserver']}
            query_fc = query_ip
        else:
            query_ip = {'name': '*%s' % name, 'scope': 'cluster'}
            query_fc = None
        fields = 'name,location,uuid,enabled,svm.name'
        fields_fc = fields + ',data_protocol'
        fields_ip = fields + ',ip,service_policy'
        if self.parameters.get('dns_domain_name'):
            fields_ip += ',dns_zone'
        if self.parameters.get('probe_port') is not None:
            fields_ip += ',probe_port'
        if self.parameters.get('is_dns_update_enabled') is not None:
            fields_ip += ',ddns_enabled'
        if self.parameters.get('subnet_name') is not None:
            fields_ip += ',subnet'
        records, error, records2, error2 = (None, None, None, None)
        if if_type in [None, 'ip']:
            records, error = self.get_interface_records_rest('ip', query_ip, fields_ip)
        if if_type in [None, 'fc'] and query_fc:
            records2, error2 = self.get_interface_records_rest('fc', query_fc, fields_fc)
        if records and records2:
            msg = 'Error fetching interface %s - found duplicate entries, please indicate interface_type.' % name
            msg += ' - ip interfaces: %s' % records
            msg += ' - fc interfaces: %s' % records2
            self.module.fail_json(msg=msg)
        if error is None and error2 is not None and records:
            error2 = None
        if error2 is None and error is not None and records2:
            error = None
        if error or error2:
            errors = [to_native(err) for err in (error, error2) if err]
            self.module.fail_json(msg='Error fetching interface details for %s: %s' % (name, ' - '.join(errors)), exception=traceback.format_exc())
        if records:
            self.set_interface_type('ip')
        if records2:
            self.set_interface_type('fc')
            records = records2
        record = self.find_exact_match(records, name) if records else None
        return self.dict_from_record(record)

    def dict_from_record(self, record):
        if not record:
            return None
        return_value = {'interface_name': record['name'], 'interface_type': self.parameters['interface_type'], 'uuid': record['uuid'], 'admin_status': 'up' if record['enabled'] else 'down'}
        if self.na_helper.safe_get(record, ['location', 'home_node', 'name']):
            return_value['home_node'] = record['location']['home_node']['name']
        if self.na_helper.safe_get(record, ['location', 'home_port', 'name']):
            return_value['home_port'] = record['location']['home_port']['name']
        if self.na_helper.safe_get(record, ['svm', 'name']):
            return_value['vserver'] = record['svm']['name']
        if 'data_protocol' in record:
            return_value['data_protocol'] = record['data_protocol']
        if 'auto_revert' in record['location']:
            return_value['is_auto_revert'] = record['location']['auto_revert']
        if 'failover' in record['location']:
            return_value['failover_scope'] = record['location']['failover']
        if self.na_helper.safe_get(record, ['ip', 'address']):
            return_value['address'] = netapp_ipaddress.validate_and_compress_ip_address(record['ip']['address'], self.module)
            if self.na_helper.safe_get(record, ['ip', 'netmask']) is not None:
                return_value['netmask'] = record['ip']['netmask']
        if self.na_helper.safe_get(record, ['service_policy', 'name']):
            return_value['service_policy'] = record['service_policy']['name']
        if self.na_helper.safe_get(record, ['location', 'node', 'name']):
            return_value['current_node'] = record['location']['node']['name']
        if self.na_helper.safe_get(record, ['location', 'port', 'name']):
            return_value['current_port'] = record['location']['port']['name']
        if self.na_helper.safe_get(record, ['dns_zone']):
            return_value['dns_domain_name'] = record['dns_zone']
        if self.na_helper.safe_get(record, ['probe_port']) is not None:
            return_value['probe_port'] = record['probe_port']
        if 'ddns_enabled' in record:
            return_value['is_dns_update_enabled'] = record['ddns_enabled']
        if self.na_helper.safe_get(record, ['subnet', 'name']):
            return_value['subnet_name'] = record['subnet']['name']
        return return_value

    def get_node_port(self, uuid):
        record, error = self.get_interface_record_rest(self.parameters['interface_type'], {'uuid': uuid}, 'location')
        if error or not record:
            return (None, None, error)
        node = self.na_helper.safe_get(record, ['location', 'node', 'name'])
        port = self.na_helper.safe_get(record, ['location', 'port', 'name'])
        return (node, port, None)

    def get_interface(self, name=None):
        """
        Return details about the interface
        :param:
            name : Name of the interface

        :return: Details about the interface. None if not found.
        :rtype: dict
        """
        if name is None:
            name = self.parameters['interface_name']
        if self.use_rest:
            return self.get_interface_rest(name)
        interface_info = netapp_utils.zapi.NaElement('net-interface-get-iter')
        interface_attributes = netapp_utils.zapi.NaElement('net-interface-info')
        interface_attributes.add_new_child('interface-name', name)
        interface_attributes.add_new_child('vserver', self.parameters['vserver'])
        query = netapp_utils.zapi.NaElement('query')
        query.add_child_elem(interface_attributes)
        interface_info.add_child_elem(query)
        try:
            result = self.server.invoke_successfully(interface_info, True)
        except netapp_utils.zapi.NaApiError as exc:
            self.module.fail_json(msg='Error fetching interface details for %s: %s' % (name, to_native(exc)), exception=traceback.format_exc())
        return_value = None
        if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
            interface_attributes = result.get_child_by_name('attributes-list').get_child_by_name('net-interface-info')
            return_value = {'interface_name': name, 'admin_status': interface_attributes['administrative-status'], 'home_port': interface_attributes['home-port'], 'home_node': interface_attributes['home-node'], 'failover_policy': interface_attributes['failover-policy'].replace('_', '-')}
            if interface_attributes.get_child_by_name('is-auto-revert'):
                return_value['is_auto_revert'] = interface_attributes['is-auto-revert'] == 'true'
            if interface_attributes.get_child_by_name('failover-group'):
                return_value['failover_group'] = interface_attributes['failover-group']
            if interface_attributes.get_child_by_name('address'):
                return_value['address'] = netapp_ipaddress.validate_and_compress_ip_address(interface_attributes['address'], self.module)
            if interface_attributes.get_child_by_name('netmask'):
                return_value['netmask'] = interface_attributes['netmask']
            if interface_attributes.get_child_by_name('firewall-policy'):
                return_value['firewall_policy'] = interface_attributes['firewall-policy']
            if interface_attributes.get_child_by_name('dns-domain-name') not in ('none', None):
                return_value['dns_domain_name'] = interface_attributes['dns-domain-name']
            else:
                return_value['dns_domain_name'] = None
            if interface_attributes.get_child_by_name('listen-for-dns-query'):
                return_value['listen_for_dns_query'] = self.na_helper.get_value_for_bool(True, interface_attributes['listen-for-dns-query'])
            if interface_attributes.get_child_by_name('is-dns-update-enabled'):
                return_value['is_dns_update_enabled'] = self.na_helper.get_value_for_bool(True, interface_attributes['is-dns-update-enabled'])
            if interface_attributes.get_child_by_name('is-ipv4-link-local'):
                return_value['is_ipv4_link_local'] = self.na_helper.get_value_for_bool(True, interface_attributes['is-ipv4-link-local'])
            if interface_attributes.get_child_by_name('service-policy'):
                return_value['service_policy'] = interface_attributes['service-policy']
            if interface_attributes.get_child_by_name('current-node'):
                return_value['current_node'] = interface_attributes['current-node']
            if interface_attributes.get_child_by_name('current-port'):
                return_value['current_port'] = interface_attributes['current-port']
        return return_value

    @staticmethod
    def set_options(options, parameters):
        """ set attributes for create or modify """
        if parameters.get('role') is not None:
            options['role'] = parameters['role']
        if parameters.get('home_node') is not None:
            options['home-node'] = parameters['home_node']
        if parameters.get('home_port') is not None:
            options['home-port'] = parameters['home_port']
        if parameters.get('subnet_name') is not None:
            options['subnet-name'] = parameters['subnet_name']
        if parameters.get('address') is not None:
            options['address'] = parameters['address']
        if parameters.get('netmask') is not None:
            options['netmask'] = parameters['netmask']
        if parameters.get('failover_policy') is not None:
            options['failover-policy'] = parameters['failover_policy']
        if parameters.get('failover_group') is not None:
            options['failover-group'] = parameters['failover_group']
        if parameters.get('firewall_policy') is not None:
            options['firewall-policy'] = parameters['firewall_policy']
        if parameters.get('is_auto_revert') is not None:
            options['is-auto-revert'] = 'true' if parameters['is_auto_revert'] else 'false'
        if parameters.get('admin_status') is not None:
            options['administrative-status'] = parameters['admin_status']
        if parameters.get('force_subnet_association') is not None:
            options['force-subnet-association'] = 'true' if parameters['force_subnet_association'] else 'false'
        if parameters.get('dns_domain_name') is not None:
            options['dns-domain-name'] = parameters['dns_domain_name']
        if parameters.get('listen_for_dns_query') is not None:
            options['listen-for-dns-query'] = 'true' if parameters['listen_for_dns_query'] else 'false'
        if parameters.get('is_dns_update_enabled') is not None:
            options['is-dns-update-enabled'] = 'true' if parameters['is_dns_update_enabled'] else 'false'
        if parameters.get('is_ipv4_link_local') is not None:
            options['is-ipv4-link-local'] = 'true' if parameters['is_ipv4_link_local'] else 'false'
        if parameters.get('service_policy') is not None:
            options['service-policy'] = parameters['service_policy']

    def fix_errors(self, options, errors):
        """ignore role and firewall_policy if a service_policy can be safely derived"""
        block_p, file_p, fcp = self.derive_block_file_type(self.parameters.get('protocols'))
        if 'role' in errors:
            fixed = False
            if errors['role'] == 'data' and errors.get('firewall_policy', 'data') == 'data':
                if fcp:
                    fixed = True
                elif file_p and self.parameters.get('service_policy', 'default-data-files') == 'default-data-files':
                    options['service_policy'] = 'default-data-files'
                    fixed = True
                elif block_p and self.parameters.get('service_policy', 'default-data-blocks') == 'default-data-blocks':
                    options['service_policy'] = 'default-data-blocks'
                    fixed = True
            if errors['role'] == 'data' and errors.get('firewall_policy') == 'mgmt':
                options['service_policy'] = 'default-management'
                fixed = True
            if errors['role'] == 'intercluster' and errors.get('firewall_policy') in [None, 'intercluster']:
                options['service_policy'] = 'default-intercluster'
                fixed = True
            if errors['role'] == 'cluster' and errors.get('firewall_policy') in [None, 'mgmt']:
                options['service_policy'] = 'default-cluster'
                fixed = True
            if errors['role'] == 'data' and fcp and (errors.get('firewall_policy') is None):
                fixed = True
            if fixed:
                errors.pop('role')
                errors.pop('firewall_policy', None)

    def set_options_rest(self, parameters):
        """ set attributes for create or modify """

        def add_ip(options, key, value):
            if 'ip' not in options:
                options['ip'] = {}
            options['ip'][key] = value

        def add_location(options, key, value, node=None):
            if 'location' not in options:
                options['location'] = {}
            if key in ['home_node', 'home_port', 'node', 'port', 'broadcast_domain']:
                options['location'][key] = {'name': value}
            else:
                options['location'][key] = value
            if key in ['home_port', 'port']:
                options['location'][key]['node'] = {'name': node}

        def get_node_for_port(parameters, pkey):
            if pkey == 'current_port':
                return parameters.get('current_node') or self.parameters.get('home_node') or self.get_home_node_for_cluster()
            elif pkey == 'home_port':
                return self.parameters.get('home_node') or self.get_home_node_for_cluster()
            else:
                return None
        options, migrate_options, errors = ({}, {}, {})
        create_with_current = False
        if parameters is None:
            parameters = self.parameters
            if self.parameters['interface_type'] == 'fc' and 'home_port' not in self.parameters:
                create_with_current = True
        mapping_params_to_rest = {'admin_status': 'enabled', 'interface_name': 'name', 'vserver': 'svm.name', 'current_port': 'port', 'home_port': 'home_port'}
        if self.parameters['interface_type'] == 'ip':
            mapping_params_to_rest.update({'ipspace': 'ipspace.name', 'service_policy': 'service_policy', 'dns_domain_name': 'dns_zone', 'is_dns_update_enabled': 'ddns_enabled', 'probe_port': 'probe_port', 'subnet_name': 'subnet.name', 'fail_if_subnet_conflicts': 'fail_if_subnet_conflicts', 'address': 'address', 'netmask': 'netmask', 'broadcast_domain': 'broadcast_domain', 'failover_scope': 'failover', 'is_auto_revert': 'auto_revert', 'home_node': 'home_node', 'current_node': 'node'})
        if self.parameters['interface_type'] == 'fc':
            mapping_params_to_rest['data_protocol'] = 'data_protocol'
        ip_keys = ('address', 'netmask')
        location_keys = ('home_port', 'home_node', 'current_port', 'current_node', 'failover_scope', 'is_auto_revert', 'broadcast_domain')
        has_home_port, has_current_port = (False, False)
        if 'home_port' in parameters:
            has_home_port = True
        if 'current_port' in parameters:
            has_current_port = True
        for pkey, rkey in mapping_params_to_rest.items():
            if pkey in parameters:
                if pkey == 'admin_status':
                    options[rkey] = parameters[pkey] == 'up'
                elif pkey in ip_keys:
                    add_ip(options, rkey, parameters[pkey])
                elif pkey in location_keys:
                    if has_home_port and pkey == 'home_node':
                        continue
                    if has_current_port and pkey == 'current_node':
                        continue
                    dest = migrate_options if rkey in ('node', 'port') and (not create_with_current) else options
                    add_location(dest, rkey, parameters[pkey], get_node_for_port(parameters, pkey))
                else:
                    options[rkey] = parameters[pkey]
        keys_in_error = ('role', 'failover_group', 'firewall_policy', 'force_subnet_association', 'listen_for_dns_query', 'is_ipv4_link_local')
        for pkey in keys_in_error:
            if pkey in parameters:
                errors[pkey] = parameters[pkey]
        return (options, migrate_options, errors)

    def set_protocol_option(self, required_keys):
        """ set protocols for create """
        if self.parameters.get('protocols') is None:
            return None
        data_protocols_obj = netapp_utils.zapi.NaElement('data-protocols')
        for protocol in self.parameters.get('protocols'):
            if protocol.lower() in ['fc-nvme', 'fcp']:
                if 'address' in required_keys:
                    required_keys.remove('address')
                if 'home_port' in required_keys:
                    required_keys.remove('home_port')
                if 'netmask' in required_keys:
                    required_keys.remove('netmask')
                not_required_params = set(['address', 'netmask', 'firewall_policy'])
                if not not_required_params.isdisjoint(set(self.parameters.keys())):
                    self.module.fail_json(msg='Error: Following parameters for creating interface are not supported for data-protocol fc-nvme: %s' % ', '.join(not_required_params))
            data_protocols_obj.add_new_child('data-protocol', protocol)
        return data_protocols_obj

    def get_cluster_node_names_rest(self):
        """ get cluster node names, but the cluster may not exist yet
            return:
                empty list if the cluster cannot be reached
                a list of nodes
        """
        if self.cluster_nodes is None:
            records, error = rest_generic.get_0_or_more_records(self.rest_api, 'cluster/nodes', fields='name,uuid,cluster_interfaces')
            if error:
                self.module.fail_json(msg='Error fetching cluster node info: %s' % to_native(error), exception=traceback.format_exc())
            self.cluster_nodes = records or []
        return [record['name'] for record in self.cluster_nodes]

    def get_home_node_for_cluster(self):
        """ get the first node name from this cluster """
        if self.use_rest:
            if not self.home_node:
                nodes = self.get_cluster_node_names_rest()
                if nodes:
                    self.home_node = nodes[0]
            return self.home_node
        get_node = netapp_utils.zapi.NaElement('cluster-node-get-iter')
        attributes = {'query': {'cluster-node-info': {}}}
        get_node.translate_struct(attributes)
        try:
            result = self.server.invoke_successfully(get_node, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as exc:
            if str(exc.code) == '13003' or exc.message == 'ZAPI is not enabled in pre-cluster mode.':
                return None
            self.module.fail_json(msg='Error fetching node for interface %s: %s' % (self.parameters['interface_name'], to_native(exc)), exception=traceback.format_exc())
        if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
            attributes = result.get_child_by_name('attributes-list')
            return attributes.get_child_by_name('cluster-node-info').get_child_content('node-name')
        return None

    def validate_rest_input_parameters(self, action=None):
        if 'vserver' in self.parameters and self.parameters.get('role') in ['cluster', 'intercluster', 'node-mgmt', 'cluster-mgmt']:
            del self.parameters['vserver']
            self.module.warn('Ignoring vserver with REST for non data SVM.')
        errors = []
        if action == 'create':
            if 'vserver' not in self.parameters and 'ipspace' not in self.parameters:
                errors.append('ipspace name must be provided if scope is cluster, or vserver for svm scope.')
            if self.parameters['interface_type'] == 'fc':
                unsupported_fc_options = ['broadcast_domain', 'dns_domain_name', 'is_dns_update_enabled', 'probe_port', 'subnet_name', 'fail_if_subnet_conflicts']
                used_unsupported_fc_options = [option for option in unsupported_fc_options if option in self.parameters]
                if used_unsupported_fc_options:
                    plural = 's' if len(used_unsupported_fc_options) > 1 else ''
                    errors.append('%s option%s only supported for IP interfaces: %s, interface_type: %s' % (', '.join(used_unsupported_fc_options), plural, self.parameters.get('interface_name'), self.parameters['interface_type']))
            if self.parameters.get('home_port') and self.parameters.get('broadcast_domain'):
                errors.append('home_port and broadcast_domain are mutually exclusive for creating: %s' % self.parameters.get('interface_name'))
        if self.parameters.get('role') == 'intercluster' and self.parameters.get('protocols') is not None:
            errors.append('Protocol cannot be specified for intercluster role, failed to create interface.')
        if errors:
            self.module.fail_json(msg='Error: %s' % '  '.join(errors))
        ignored_keys = []
        for key in self.parameters.get('ignore_zapi_options', []):
            if key in self.parameters:
                del self.parameters[key]
                ignored_keys.append(key)
        if ignored_keys:
            self.module.warn('Ignoring %s' % ', '.join(ignored_keys))

    def validate_required_parameters(self, keys):
        """
            Validate if required parameters for create or modify are present.
            Parameter requirement might vary based on given data-protocol.
            :return: None
        """
        home_node = self.parameters.get('home_node') or self.get_home_node_for_cluster()
        errors = []
        if self.use_rest and home_node is None and (self.parameters.get('home_port') is not None):
            errors.append('Cannot guess home_node, home_node is required when home_port is present with REST.')
        if 'broadcast_domain_home_port_or_home_node' in keys:
            if all((x not in self.parameters for x in ['broadcast_domain', 'home_port', 'home_node'])):
                errors.append("At least one of 'broadcast_domain', 'home_port', 'home_node' is required to create an IP interface.")
            keys.remove('broadcast_domain_home_port_or_home_node')
        if not keys.issubset(set(self.parameters.keys())):
            errors.append('Missing one or more required parameters for creating interface: %s.' % ', '.join(keys))
        if 'interface_type' in keys and 'interface_type' in self.parameters:
            if self.parameters['interface_type'] not in ['fc', 'ip']:
                errors.append('unexpected value for interface_type: %s.' % self.parameters['interface_type'])
            elif self.parameters['interface_type'] == 'fc':
                if not self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 8, 0):
                    if 'home_port' in self.parameters:
                        errors.append("'home_port' is not supported for FC interfaces with 9.7, use 'current_port', avoid home_node.")
                    if 'home_node' in self.parameters:
                        self.module.warn("Avoid 'home_node' with FC interfaces with 9.7, use 'current_node'.")
                if 'vserver' not in self.parameters:
                    errors.append("A data 'vserver' is required for FC interfaces.")
                if 'service_policy' in self.parameters:
                    errors.append("'service_policy' is not supported for FC interfaces.")
                if 'role' in self.parameters and self.parameters.get('role') != 'data':
                    errors.append("'role' is deprecated, and 'data' is the only value supported for FC interfaces: found %s." % self.parameters.get('role'))
                if 'probe_port' in self.parameters:
                    errors.append("'probe_port' is not supported for FC interfaces.")
        if errors:
            self.module.fail_json(msg='Error: %s' % '  '.join(errors))

    def validate_modify_parameters(self, body):
        """ Only the following keys can be modified:
            enabled, ip, location, name, service_policy
        """
        bad_keys = [key for key in body if key not in ['enabled', 'ip', 'location', 'name', 'service_policy', 'dns_zone', 'ddns_enabled', 'subnet.name', 'fail_if_subnet_conflicts']]
        if bad_keys:
            plural = 's' if len(bad_keys) > 1 else ''
            self.module.fail_json(msg='The following option%s cannot be modified: %s' % (plural, ', '.join(bad_keys)))

    def build_rest_body(self, modify=None):
        required_keys = set(['interface_type'])
        self.validate_required_parameters(required_keys)
        self.validate_rest_input_parameters(action='modify' if modify else 'create')
        if modify:
            if self.parameters.get('fail_if_subnet_conflicts') is not None:
                modify['fail_if_subnet_conflicts'] = self.parameters['fail_if_subnet_conflicts']
        else:
            required_keys = set()
            required_keys.add('interface_name')
            if self.parameters['interface_type'] == 'fc':
                self.derive_fc_data_protocol()
                required_keys.add('data_protocol')
                if 'home_port' not in self.parameters:
                    if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 8, 0):
                        required_keys.add('home_port')
                    else:
                        required_keys.add('current_port')
            if self.parameters['interface_type'] == 'ip':
                if 'subnet_name' not in self.parameters:
                    required_keys.add('address')
                    required_keys.add('netmask')
                required_keys.add('broadcast_domain_home_port_or_home_node')
            self.validate_required_parameters(required_keys)
        body, migrate_body, errors = self.set_options_rest(modify)
        self.fix_errors(body, errors)
        if errors:
            self.module.fail_json(msg='Error %s interface, unsupported options: %s' % ('modifying' if modify else 'creating', str(errors)))
        if modify:
            self.validate_modify_parameters(body)
        return (body, migrate_body)

    def create_interface_rest(self, body):
        """ calling REST to create interface """
        query = {'return_records': 'true'}
        records, error = rest_generic.post_async(self.rest_api, self.get_net_int_api(), body, query)
        if error:
            self.module.fail_json(msg='Error creating interface %s: %s' % (self.parameters['interface_name'], to_native(error)), exception=traceback.format_exc())
        return records

    def create_interface(self, body):
        """ calling zapi to create interface """
        if self.use_rest:
            return self.create_interface_rest(body)
        required_keys = set(['role', 'home_port'])
        if self.parameters.get('subnet_name') is None and self.parameters.get('is_ipv4_link_local') is None:
            required_keys.add('address')
            required_keys.add('netmask')
        if self.parameters.get('service_policy') is not None:
            required_keys.remove('role')
        data_protocols_obj = self.set_protocol_option(required_keys)
        self.validate_required_parameters(required_keys)
        options = {'interface-name': self.parameters['interface_name'], 'vserver': self.parameters['vserver']}
        NetAppOntapInterface.set_options(options, self.parameters)
        interface_create = netapp_utils.zapi.NaElement.create_node_with_children('net-interface-create', **options)
        if data_protocols_obj is not None:
            interface_create.add_child_elem(data_protocols_obj)
        try:
            self.server.invoke_successfully(interface_create, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as exc:
            if to_native(exc.code) == '17':
                self.na_helper.changed = False
            else:
                self.module.fail_json(msg='Error Creating interface %s: %s' % (self.parameters['interface_name'], to_native(exc)), exception=traceback.format_exc())

    def delete_interface_rest(self, uuid):
        """ calling zapi to delete interface """
        dummy, error = rest_generic.delete_async(self.rest_api, self.get_net_int_api(), uuid)
        if error:
            self.module.fail_json(msg='Error deleting interface %s: %s' % (self.parameters['interface_name'], to_native(error)), exception=traceback.format_exc())

    def delete_interface(self, current_status, current_interface, uuid):
        """ calling zapi to delete interface """
        if current_status == 'up':
            self.parameters['admin_status'] = 'down'
            if self.use_rest:
                if current_interface == 'fc':
                    self.modify_interface_rest(uuid, {'enabled': False})
            else:
                self.modify_interface({'admin_status': 'down'})
        if self.use_rest:
            return self.delete_interface_rest(uuid)
        interface_delete = netapp_utils.zapi.NaElement.create_node_with_children('net-interface-delete', **{'interface-name': self.parameters['interface_name'], 'vserver': self.parameters['vserver']})
        try:
            self.server.invoke_successfully(interface_delete, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as exc:
            self.module.fail_json(msg='Error deleting interface %s: %s' % (self.parameters['interface_name'], to_native(exc)), exception=traceback.format_exc())

    def modify_interface_rest(self, uuid, body):
        """ calling REST to modify interface """
        if not body:
            return
        dummy, error = rest_generic.patch_async(self.rest_api, self.get_net_int_api(), uuid, body)
        if error:
            self.module.fail_json(msg='Error modifying interface %s: %s' % (self.parameters['interface_name'], to_native(error)), exception=traceback.format_exc())

    def migrate_interface_rest(self, uuid, body):
        errors = []
        desired_node = self.na_helper.safe_get(body, ['location', 'node', 'name'])
        desired_port = self.na_helper.safe_get(body, ['location', 'port', 'name'])
        for __ in range(12):
            self.modify_interface_rest(uuid, body)
            time.sleep(10)
            node, port, error = self.get_node_port(uuid)
            if error is None and desired_node in [None, node] and (desired_port in [None, port]):
                return
            if errors or error is not None:
                errors.append(str(error))
        if errors:
            self.module.fail_json(msg='Errors waiting for migration to complete: %s' % ' - '.join(errors))
        else:
            self.module.warn('Failed to confirm interface is migrated after 120 seconds')

    def modify_interface(self, modify, uuid=None, body=None):
        """
        Modify the interface.
        """
        if self.use_rest:
            return self.modify_interface_rest(uuid, body)
        migrate = {}
        modify_options = dict(modify)
        if modify_options.get('current_node') is not None:
            migrate['current_node'] = modify_options.pop('current_node')
        if modify_options.get('current_port') is not None:
            migrate['current_port'] = modify_options.pop('current_port')
        if modify_options:
            options = {'interface-name': self.parameters['interface_name'], 'vserver': self.parameters['vserver']}
            NetAppOntapInterface.set_options(options, modify_options)
            interface_modify = netapp_utils.zapi.NaElement.create_node_with_children('net-interface-modify', **options)
            try:
                self.server.invoke_successfully(interface_modify, enable_tunneling=True)
            except netapp_utils.zapi.NaApiError as err:
                self.module.fail_json(msg='Error modifying interface %s: %s' % (self.parameters['interface_name'], to_native(err)), exception=traceback.format_exc())
        if migrate:
            self.migrate_interface()

    def migrate_interface(self):
        interface_migrate = netapp_utils.zapi.NaElement('net-interface-migrate')
        if self.parameters.get('current_node') is None:
            self.module.fail_json(msg='current_node must be set to migrate')
        interface_migrate.add_new_child('destination-node', self.parameters['current_node'])
        if self.parameters.get('current_port') is not None:
            interface_migrate.add_new_child('destination-port', self.parameters['current_port'])
        interface_migrate.add_new_child('lif', self.parameters['interface_name'])
        interface_migrate.add_new_child('vserver', self.parameters['vserver'])
        try:
            self.server.invoke_successfully(interface_migrate, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error migrating %s: %s' % (self.parameters['current_node'], to_native(error)), exception=traceback.format_exc())
        try:
            self.server.invoke_successfully(interface_migrate, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error migrating %s: %s' % (self.parameters['current_node'], to_native(error)), exception=traceback.format_exc())

    def rename_interface(self):
        options = {'interface-name': self.parameters['from_name'], 'new-name': self.parameters['interface_name'], 'vserver': self.parameters['vserver']}
        interface_rename = netapp_utils.zapi.NaElement.create_node_with_children('net-interface-rename', **options)
        try:
            self.server.invoke_successfully(interface_rename, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error renaming %s to %s: %s' % (self.parameters['from_name'], self.parameters['interface_name'], to_native(error)), exception=traceback.format_exc())

    def get_action(self):
        modify, rename, new_name = (None, None, None)
        current = self.get_interface()
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if cd_action == 'create' and self.parameters.get('from_name'):
            new_name = self.parameters['interface_name']
            old_interface = self.get_interface(self.parameters['from_name'])
            rename = self.na_helper.is_rename_action(old_interface, current)
            if rename is None:
                self.module.fail_json(msg='Error renaming interface %s: no interface with from_name %s.' % (self.parameters['interface_name'], self.parameters['from_name']))
            if rename:
                current = old_interface
                cd_action = None
        if cd_action is None:
            modify = self.na_helper.get_modified_attributes(current, self.parameters)
        if rename and self.use_rest:
            rename = False
            if 'interface_name' not in modify:
                modify['interface_name'] = new_name
        if modify and modify.get('home_node') == 'localhost':
            modify.pop('home_node')
            if not modify:
                self.na_helper.changed = False
        return (cd_action, modify, rename, current)

    def build_rest_payloads(self, cd_action, modify, current):
        body, migrate_body = (None, None)
        uuid = current.get('uuid') if current else None
        if self.use_rest:
            if cd_action == 'create':
                body, migrate_body = self.build_rest_body()
            elif modify:
                if modify.get('home_node') and (not modify.get('home_port')) and (self.parameters['interface_type'] == 'fc'):
                    modify['home_port'] = current['home_port']
                if modify.get('current_node') and (not modify.get('current_port')) and (self.parameters['interface_type'] == 'fc'):
                    modify['current_port'] = current['current_port']
                body, migrate_body = self.build_rest_body(modify)
            if (modify or cd_action == 'delete') and uuid is None:
                self.module.fail_json(msg='Error, expecting uuid in existing record')
            desired_home_port = self.na_helper.safe_get(body, ['location', 'home_port'])
            desired_current_port = self.na_helper.safe_get(migrate_body, ['location', 'port'])
            if self.parameters.get('interface_type') == 'fc' and desired_home_port and desired_current_port and (desired_home_port == desired_current_port):
                migrate_body = None
        return (uuid, body, migrate_body)

    def apply(self):
        """ calling all interface features """
        cd_action, modify, rename, current = self.get_action()
        uuid, body, migrate_body = self.build_rest_payloads(cd_action, modify, current)
        if self.na_helper.changed and (not self.module.check_mode):
            if rename and (not self.use_rest):
                self.rename_interface()
                modify.pop('interface_name')
            if cd_action == 'create':
                records = self.create_interface(body)
                if records:
                    uuid = records['records'][0]['uuid']
            elif cd_action == 'delete':
                interface_type = current['interface_type'] if self.use_rest else None
                self.delete_interface(current['admin_status'], interface_type, uuid)
            elif modify:
                self.modify_interface(modify, uuid, body)
            if migrate_body:
                if self.parameters.get('interface_type') == 'fc' and self.use_rest and self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 8, 0):
                    self.module.fail_json(msg='Error: cannot migrate FC interface')
                self.migrate_interface_rest(uuid, migrate_body)
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify)
        self.module.exit_json(**result)