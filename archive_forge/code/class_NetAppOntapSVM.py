from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver, zapis_svm
class NetAppOntapSVM:
    """ create, delete, modify, rename SVM (aka vserver) """

    def __init__(self):
        self.use_rest = False
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), name=dict(required=True, type='str', aliases=['vserver']), from_name=dict(required=False, type='str'), admin_state=dict(required=False, type='str', choices=['running', 'stopped']), root_volume=dict(type='str'), root_volume_aggregate=dict(type='str'), root_volume_security_style=dict(type='str', choices=['unix', 'ntfs', 'mixed', 'unified']), allowed_protocols=dict(type='list', elements='str'), aggr_list=dict(type='list', elements='str'), ipspace=dict(type='str', required=False), snapshot_policy=dict(type='str', required=False), language=dict(type='str', required=False), subtype=dict(type='str', choices=['default', 'dp_destination', 'sync_source', 'sync_destination']), comment=dict(type='str', required=False), ignore_rest_unsupported_options=dict(type='bool', default=False), max_volumes=dict(type='str'), services=dict(type='dict', options=dict(cifs=dict(type='dict', options=dict(allowed=dict(type='bool'))), iscsi=dict(type='dict', options=dict(allowed=dict(type='bool'), enabled=dict(type='bool'))), fcp=dict(type='dict', options=dict(allowed=dict(type='bool'), enabled=dict(type='bool'))), nfs=dict(type='dict', options=dict(allowed=dict(type='bool'), enabled=dict(type='bool'))), nvme=dict(type='dict', options=dict(allowed=dict(type='bool'), enabled=dict(type='bool'))), ndmp=dict(type='dict', options=dict(allowed=dict(type='bool'))))), web=dict(type='dict', options=dict(certificate=dict(type='str'), client_enabled=dict(type='bool'), ocsp_enabled=dict(type='bool')))))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True, mutually_exclusive=[('allowed_protocols', 'services'), ('services', 'root_volume'), ('services', 'root_volume_aggregate'), ('services', 'root_volume_security_style')])
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        if 'language' in self.parameters and self.parameters['language'].lower() == 'c.utf-8':
            self.parameters['language'] = 'c.utf_8'
        self.rest_api = OntapRestAPI(self.module)
        self.timeout = self.rest_api.timeout
        self.allowable_protocols_rest = netapp_utils.get_feature(self.module, 'svm_allowable_protocols_rest')
        self.allowable_protocols_zapi = netapp_utils.get_feature(self.module, 'svm_allowable_protocols_zapi')
        self.use_rest = self.validate_options()
        if not self.use_rest:
            if not netapp_utils.has_netapp_lib():
                self.module.fail_json(msg=netapp_utils.netapp_lib_is_required())
            self.server = netapp_utils.setup_na_ontap_zapi(module=self.module)
            if self.parameters.get('admin_state') is not None:
                self.parameters.pop('admin_state')
                self.module.warn('admin_state is ignored when ZAPI is used.')

    def validate_int_or_string(self, value, astring):
        if value is None or value == astring:
            return
        try:
            int_value = int(value)
        except ValueError:
            int_value = None
        if int_value is None or str(int_value) != value:
            self.module.fail_json(msg="Error: expecting int value or '%s', got: %s - %s" % (astring, value, int_value))

    def validate_options(self):
        unsupported_rest_properties = ['root_volume', 'root_volume_aggregate', 'root_volume_security_style']
        required_unsupported_rest_properties = [] if self.parameters['ignore_rest_unsupported_options'] else unsupported_rest_properties
        ignored_unsupported_rest_properties = unsupported_rest_properties if self.parameters['ignore_rest_unsupported_options'] else []
        used_required_unsupported_rest_properties = [x for x in required_unsupported_rest_properties if x in self.parameters]
        used_ignored_unsupported_rest_properties = [x for x in ignored_unsupported_rest_properties if x in self.parameters]
        use_rest, error = self.rest_api.is_rest(used_required_unsupported_rest_properties)
        if error is not None:
            self.module.fail_json(msg=error)
        if use_rest and used_ignored_unsupported_rest_properties:
            self.module.warn('Using REST and ignoring: %s' % used_ignored_unsupported_rest_properties)
            for attr in used_ignored_unsupported_rest_properties:
                del self.parameters[attr]
        if use_rest and 'aggr_list' in self.parameters and (self.parameters['aggr_list'] == ['*']):
            self.module.warn("Using REST and ignoring aggr_list: '*'")
            del self.parameters['aggr_list']
        if use_rest and self.parameters.get('allowed_protocols') is not None:
            self.parameters['services'] = dict(((protocol, {'allowed': protocol in self.parameters['allowed_protocols']}) for protocol in self.allowable_protocols_rest))
        if self.parameters.get('allowed_protocols'):
            allowable = self.allowable_protocols_rest if use_rest else self.allowable_protocols_zapi
            errors = ['Unexpected value %s in allowed_protocols.' % protocol for protocol in self.parameters['allowed_protocols'] if protocol not in allowable]
            if errors:
                self.module.fail_json(msg='Error - %s' % '  '.join(errors))
        if use_rest and self.parameters.get('services') and (not self.parameters.get('allowed_protocols')):
            if self.parameters['services'].get('ndmp') and (not self.rest_api.meets_rest_minimum_version(use_rest, 9, 10, 1)):
                self.module.fail_json(msg=self.rest_api.options_require_ontap_version('ndmp', '9.10.1', use_rest=use_rest))
        if self.parameters.get('services') and (not use_rest):
            self.module.fail_json(msg=self.rest_api.options_require_ontap_version('services', use_rest=use_rest))
        if self.parameters.get('web'):
            if not use_rest or not self.rest_api.meets_rest_minimum_version(use_rest, 9, 8, 0):
                self.module.fail_json(msg=self.rest_api.options_require_ontap_version('web', '9.8', use_rest=use_rest))
            if not self.rest_api.meets_rest_minimum_version(use_rest, 9, 10, 1):
                suboptions = ('client_enabled', 'ocsp_enabled')
                for suboption in suboptions:
                    if self.parameters['web'].get(suboption) is not None:
                        self.module.fail_json(msg=self.rest_api.options_require_ontap_version(suboptions, '9.10.1', use_rest=use_rest))
            if self.parameters['web'].get('certificate'):
                self.parameters['web']['certificate'] = {'name': self.parameters['web']['certificate']}
                self.set_certificate_uuid()
        self.validate_int_or_string(self.parameters.get('max_volumes'), 'unlimited')
        return use_rest

    def clean_up_output(self, vserver_details):
        vserver_details['root_volume'] = None
        vserver_details['root_volume_aggregate'] = None
        vserver_details['root_volume_security_style'] = None
        vserver_details['aggr_list'] = [aggr['name'] for aggr in vserver_details['aggregates']]
        vserver_details.pop('aggregates')
        vserver_details['ipspace'] = vserver_details['ipspace']['name']
        vserver_details['snapshot_policy'] = vserver_details['snapshot_policy']['name']
        vserver_details['admin_state'] = vserver_details.pop('state')
        if 'max_volumes' in vserver_details:
            vserver_details['max_volumes'] = str(vserver_details['max_volumes'])
        if vserver_details.get('web') is None and self.parameters.get('web'):
            vserver_details['web'] = {'certificate': {'uuid': vserver_details['certificate']['uuid'] if 'certificate' in vserver_details else None}, 'client_enabled': None, 'ocsp_enabled': None}
        services = {}
        allowed_protocols = None if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 9, 1) else vserver_details.get('allowed_protocols')
        for protocol in self.allowable_protocols_rest:
            allowed = self.na_helper.safe_get(vserver_details, [protocol, 'allowed'])
            if allowed is None and allowed_protocols is not None:
                allowed = protocol in allowed_protocols
            enabled = self.na_helper.safe_get(vserver_details, [protocol, 'enabled'])
            if allowed is not None or enabled is not None:
                services[protocol] = {}
            if allowed is not None:
                services[protocol]['allowed'] = allowed
            if enabled is not None:
                services[protocol]['enabled'] = enabled
        if services:
            vserver_details['services'] = services
        return vserver_details

    def get_certificates(self, cert_type):
        """Retrieve list of certificates"""
        api = 'security/certificates'
        query = {'svm.name': self.parameters['name'], 'type': cert_type}
        records, error = rest_generic.get_0_or_more_records(self.rest_api, api, query)
        if error:
            self.module.fail_json(msg='Error retrieving certificates: %s' % error)
        return [record['name'] for record in records] if records else []

    def set_certificate_uuid(self):
        """Retrieve certicate uuid for 9.8 or later"""
        api = 'security/certificates'
        query = {'name': self.parameters['web']['certificate']['name'], 'svm.name': self.parameters['name'], 'type': 'server'}
        record, error = rest_generic.get_one_record(self.rest_api, api, query)
        if error:
            self.module.fail_json(msg='Error retrieving certificate %s: %s' % (self.parameters['web']['certificate'], error))
        if not record:
            self.module.fail_json(msg='Error certificate not found: %s.  Current certificates with type=server: %s' % (self.parameters['web']['certificate'], self.get_certificates('server')))
        self.parameters['web']['certificate']['uuid'] = record['uuid']

    def get_web_service(self, uuid):
        """Retrieve web service info for 9.10.1 or later"""
        api = 'svm/svms/%s/web' % uuid
        record, error = rest_generic.get_one_record(self.rest_api, api)
        if error:
            self.module.fail_json(msg='Error retrieving web info: %s' % error)
        return record

    def get_vserver(self, vserver_name=None):
        """
        Checks if vserver exists.

        :return:
            vserver object if vserver found
            None if vserver is not found
        :rtype: object/None
        """
        if vserver_name is None:
            vserver_name = self.parameters['name']
        if self.use_rest:
            fields = 'subtype,aggregates,language,snapshot_policy,ipspace,comment,nfs,cifs,fcp,iscsi,nvme,state'
            if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 9, 1):
                fields += ',max_volumes'
            if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 8, 0) and (not self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 10, 1)):
                fields += ',certificate'
            if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 10, 1):
                fields += ',ndmp'
            record, error = rest_vserver.get_vserver(self.rest_api, vserver_name, fields)
            if error:
                self.module.fail_json(msg=error)
            if record:
                if self.parameters.get('web') and self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 9, 1):
                    record['web'] = self.get_web_service(record['uuid'])
                if not self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 9, 1):
                    record['allowed_protocols'], max_volumes = self.get_allowed_protocols_and_max_volumes()
                    if self.parameters.get('max_volumes') is not None:
                        record['max_volumes'] = max_volumes
                return self.clean_up_output(copy.deepcopy(record))
            return None
        return zapis_svm.get_vserver(self.server, vserver_name)

    def create_vserver(self):
        if self.use_rest:
            self.create_vserver_rest()
        else:
            options = {'vserver-name': self.parameters['name']}
            self.add_parameter_to_dict(options, 'root_volume', 'root-volume')
            self.add_parameter_to_dict(options, 'root_volume_aggregate', 'root-volume-aggregate')
            self.add_parameter_to_dict(options, 'root_volume_security_style', 'root-volume-security-style')
            self.add_parameter_to_dict(options, 'language', 'language')
            self.add_parameter_to_dict(options, 'ipspace', 'ipspace')
            self.add_parameter_to_dict(options, 'snapshot_policy', 'snapshot-policy')
            self.add_parameter_to_dict(options, 'subtype', 'vserver-subtype')
            self.add_parameter_to_dict(options, 'comment', 'comment')
            vserver_create = netapp_utils.zapi.NaElement.create_node_with_children('vserver-create', **options)
            try:
                self.server.invoke_successfully(vserver_create, enable_tunneling=False)
            except netapp_utils.zapi.NaApiError as exc:
                self.module.fail_json(msg='Error provisioning SVM %s: %s' % (self.parameters['name'], to_native(exc)), exception=traceback.format_exc())
            options = dict(((key, self.parameters[key]) for key in ('allowed_protocols', 'aggr_list', 'max_volumes') if self.parameters.get(key)))
            if options:
                self.modify_vserver(options)

    def create_body_contents(self, modify=None):
        keys_to_modify = self.parameters.keys() if modify is None else modify.keys()
        protocols_to_modify = self.parameters.get('services', {}) if modify is None else modify.get('services', {})
        simple_keys = ['name', 'language', 'ipspace', 'snapshot_policy', 'subtype', 'comment']
        if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 9, 1):
            simple_keys.append('max_volumes')
        body = dict(((key, self.parameters[key]) for key in simple_keys if self.parameters.get(key) and key in keys_to_modify))
        if modify and 'admin_state' in keys_to_modify:
            body['state'] = self.parameters['admin_state']
        if 'aggr_list' in keys_to_modify:
            body['aggregates'] = [{'name': aggr} for aggr in self.parameters['aggr_list']]
        if 'certificate' in keys_to_modify:
            body['certificate'] = modify['certificate']
        allowed_protocols = {}
        for protocol, config in protocols_to_modify.items():
            if not config:
                continue
            acopy = self.na_helper.filter_out_none_entries(config)
            if modify is not None:
                acopy.pop('enabled', None)
            if not self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 9, 1):
                allowed = acopy.pop('allowed', None)
                if allowed is not None:
                    allowed_protocols[protocol] = allowed
            if acopy:
                body[protocol] = acopy
        return (body, allowed_protocols)

    def get_allowed_protocols_and_max_volumes(self):
        query = {'vserver': self.parameters['name']}
        fields = 'allowed_protocols'
        if self.parameters.get('max_volumes') is not None:
            fields += ',max_volumes'
        response, error = rest_generic.get_one_record(self.rest_api, 'private/cli/vserver', query, fields)
        if error:
            self.module.fail_json(msg='Error getting vserver info: %s - %s' % (error, response))
        if response and 'max_volumes' in response:
            max_volumes = str(response['max_volumes'])
        allowed_protocols, max_volumes = ([], None)
        if response and 'allowed_protocols' in response:
            allowed_protocols = response['allowed_protocols']
        if response and 'max_volumes' in response:
            max_volumes = str(response['max_volumes'])
        return (allowed_protocols, max_volumes)

    def rest_cli_set_max_volumes(self):
        query = {'vserver': self.parameters['name']}
        body = {'max_volumes': self.parameters['max_volumes']}
        response, error = rest_generic.patch_async(self.rest_api, 'private/cli/vserver', None, body, query)
        if error:
            self.module.fail_json(msg='Error updating max_volumes: %s - %s' % (error, response))

    def rest_cli_add_remove_protocols(self, protocols):
        protocols_to_add = [protocol for protocol, value in protocols.items() if value]
        if protocols_to_add:
            self.rest_cli_add_protocols(protocols_to_add)
        protocols_to_delete = [protocol for protocol, value in protocols.items() if not value]
        if protocols_to_delete:
            self.rest_cli_remove_protocols(protocols_to_delete)

    def rest_cli_add_protocols(self, protocols):
        query = {'vserver': self.parameters['name']}
        body = {'protocols': protocols}
        response, error = rest_generic.patch_async(self.rest_api, 'private/cli/vserver/add-protocols', None, body, query)
        if error:
            self.module.fail_json(msg='Error adding protocols: %s - %s' % (error, response))

    def rest_cli_remove_protocols(self, protocols):
        query = {'vserver': self.parameters['name']}
        body = {'protocols': protocols}
        response, error = rest_generic.patch_async(self.rest_api, 'private/cli/vserver/remove-protocols', None, body, query)
        if error:
            self.module.fail_json(msg='Error removing protocols: %s - %s' % (error, response))

    def create_vserver_rest(self):
        body, allowed_protocols = self.create_body_contents()
        dummy, error = rest_generic.post_async(self.rest_api, 'svm/svms', body, timeout=self.timeout)
        if error:
            self.module.fail_json(msg='Error in create: %s' % error)
        if self.parameters.get('max_volumes') is not None and (not self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 9, 1)):
            self.rest_cli_set_max_volumes()
        if allowed_protocols:
            self.rest_cli_add_remove_protocols(allowed_protocols)

    def delete_vserver(self, current=None):
        if self.use_rest:
            if current is None:
                self.module.fail_json(msg='Internal error, expecting SVM object in delete')
            dummy, error = rest_generic.delete_async(self.rest_api, 'svm/svms', current['uuid'], timeout=self.timeout)
            if error:
                self.module.fail_json(msg='Error in delete: %s' % error)
        else:
            vserver_delete = netapp_utils.zapi.NaElement.create_node_with_children('vserver-destroy', **{'vserver-name': self.parameters['name']})
            try:
                self.server.invoke_successfully(vserver_delete, enable_tunneling=False)
            except netapp_utils.zapi.NaApiError as exc:
                self.module.fail_json(msg='Error deleting SVM %s: %s' % (self.parameters['name'], to_native(exc)), exception=traceback.format_exc())

    def rename_vserver(self):
        """ ZAPI only, for REST it is handled as a modify"""
        vserver_rename = netapp_utils.zapi.NaElement.create_node_with_children('vserver-rename', **{'vserver-name': self.parameters['from_name'], 'new-name': self.parameters['name']})
        try:
            self.server.invoke_successfully(vserver_rename, enable_tunneling=False)
        except netapp_utils.zapi.NaApiError as exc:
            self.module.fail_json(msg='Error renaming SVM %s: %s' % (self.parameters['from_name'], to_native(exc)), exception=traceback.format_exc())

    def modify_vserver(self, modify, current=None):
        """
        Modify vserver.
        :param modify: list of modify attributes
        :param current: with rest, SVM object to modify
        """
        if self.use_rest:
            if current is None:
                self.module.fail_json(msg='Internal error, expecting SVM object in modify.')
            if not modify:
                self.module.fail_json(msg='Internal error, expecting something to modify in modify.')
            if 'name' in modify:
                body = {'name': modify['name']}
                dummy, error = rest_generic.patch_async(self.rest_api, 'svm/svms', current['uuid'], body, timeout=self.timeout)
                if error:
                    self.module.fail_json(msg='Error in rename: %s' % error, modify=modify)
                del modify['name']
            if 'web' in modify and (not self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 10, 1)):
                uuid = self.na_helper.safe_get(modify, ['web', 'certificate', 'uuid'])
                if uuid:
                    modify['certificate'] = {'uuid': uuid}
                modify.pop('web')
            body, allowed_protocols = self.create_body_contents(modify)
            if body:
                dummy, error = rest_generic.patch_async(self.rest_api, 'svm/svms', current['uuid'], body, timeout=self.timeout)
                if error:
                    self.module.fail_json(msg='Error in modify: %s' % error, modify=modify)
            if 'max_volumes' in modify and (not self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 9, 1)):
                self.rest_cli_set_max_volumes()
            if allowed_protocols:
                self.rest_cli_add_remove_protocols(allowed_protocols)
            if 'services' in modify:
                self.modify_services(modify, current)
            if 'web' in modify and self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 10, 1):
                self.modify_web_services(modify['web'], current)
        else:
            zapis_svm.modify_vserver(self.server, self.module, self.parameters['name'], modify, self.parameters)

    def modify_services(self, modify, current):
        apis = {'fcp': 'protocols/san/fcp/services', 'iscsi': 'protocols/san/iscsi/services', 'nfs': 'protocols/nfs/services', 'nvme': 'protocols/nvme/services', 'ndmp': 'protocols/ndmp/svms'}
        for protocol, config in modify['services'].items():
            enabled = config.get('enabled')
            if enabled is None:
                continue
            api = apis.get(protocol)
            if not api:
                self.module.fail_json(msg='Internal error, unexpecting service: %s.' % protocol)
            if enabled:
                link = self.na_helper.safe_get(current, [protocol, '_links', 'self', 'href'])
            body = {'enabled': enabled}
            if enabled and (not link):
                body['svm.name'] = self.parameters['name']
                dummy, error = rest_generic.post_async(self.rest_api, api, body)
            else:
                dummy, error = rest_generic.patch_async(self.rest_api, api, current['uuid'], body)
            if error:
                self.module.fail_json(msg='Error in modify service for %s: %s' % (protocol, error))

    def modify_web_services(self, record, current):
        """Patch web service for 9.10.1 or later"""
        api = 'svm/svms/%s/web' % current['uuid']
        if 'certificate' in record:
            record['certificate'].pop('name', None)
        body = self.na_helper.filter_out_none_entries(copy.deepcopy(record))
        if not body:
            self.module.warn('Nothing to change: %s' % record)
            return
        dummy, error = rest_generic.patch_async(self.rest_api, api, None, body)
        if error:
            self.module.fail_json(msg='Error in modify web service for %s: %s' % (body, error))

    def add_parameter_to_dict(self, adict, name, key=None, tostr=False):
        """
        add defined parameter (not None) to adict using key.
        :param adict: a dictionary.
        :param name: name in self.parameters.
        :param key:  key in adict.
        :param tostr: boolean.
        """
        if key is None:
            key = name
        if self.parameters.get(name) is not None:
            if tostr:
                adict[key] = str(self.parameters.get(name))
            else:
                adict[key] = self.parameters.get(name)

    def warn_when_possible_language_match(self, desired, current):
        transformed = desired.lower().replace('-', '_')
        if transformed == current:
            self.module.warn('Attempting to change language from ONTAP value %s to %s.  Use %s to suppress this warning and maintain idempotency.' % (current, desired, current))

    def apply(self):
        """Call create/modify/delete operations."""
        current = self.get_vserver()
        cd_action, rename = (None, None)
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if cd_action == 'create' and self.parameters.get('from_name'):
            old_svm = self.get_vserver(self.parameters['from_name'])
            rename = self.na_helper.is_rename_action(old_svm, current)
            if rename is None:
                self.module.fail_json(msg='Error renaming SVM %s: no SVM with from_name %s.' % (self.parameters['name'], self.parameters['from_name']))
            if rename:
                current = old_svm
                cd_action = None
        modify = self.na_helper.get_modified_attributes(current, self.parameters) if cd_action is None else {}
        if 'language' in modify:
            self.warn_when_possible_language_match(modify['language'], current['language'])
        fixed_attributes = ['root_volume', 'root_volume_aggregate', 'root_volume_security_style', 'subtype', 'ipspace']
        msgs = ['%s - current: %s - desired: %s' % (attribute, current[attribute], self.parameters[attribute]) for attribute in fixed_attributes if attribute in modify]
        if msgs:
            self.module.fail_json(msg='Error modifying SVM %s: cannot modify %s.' % (self.parameters['name'], ', '.join(msgs)))
        if self.na_helper.changed and (not self.module.check_mode):
            if rename:
                if self.use_rest:
                    modify['name'] = self.parameters['name']
                else:
                    self.rename_vserver()
                    modify.pop('name', None)
            if cd_action == 'create':
                self.create_vserver()
                if self.parameters.get('admin_state') == 'stopped':
                    current = self.get_vserver()
                    modify = {'admin_state': 'stopped'}
            elif cd_action == 'delete':
                self.delete_vserver(current)
            if modify:
                self.modify_vserver(modify, current)
            if modify and 'aggr_list' in modify and ('*' in modify['aggr_list']):
                self.module.warn("na_ontap_svm: changed always 'True' when aggr_list is '*'.")
        results = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify)
        self.module.exit_json(**results)