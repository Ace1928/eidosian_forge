from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppOntapcifsServer:
    """
    object to describe  cifs_server info
    """

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, choices=['present', 'absent'], default='present'), service_state=dict(required=False, choices=['stopped', 'started']), name=dict(required=True, type='str', aliases=['cifs_server_name']), workgroup=dict(required=False, type='str', default=None), domain=dict(required=False, type='str'), admin_user_name=dict(required=False, type='str'), admin_password=dict(required=False, type='str', no_log=True), ou=dict(required=False, type='str'), default_site=dict(required=False, type='str'), force=dict(required=False, type='bool'), vserver=dict(required=True, type='str'), from_name=dict(required=False, type='str'), smb_signing=dict(required=False, type='bool'), encrypt_dc_connection=dict(required=False, type='bool'), kdc_encryption=dict(required=False, type='bool'), smb_encryption=dict(required=False, type='bool'), restrict_anonymous=dict(required=False, type='str', choices=['no_enumeration', 'no_restriction', 'no_access']), aes_netlogon_enabled=dict(required=False, type='bool'), ldap_referral_enabled=dict(required=False, type='bool'), session_security=dict(required=False, type='str', choices=['none', 'sign', 'seal']), lm_compatibility_level=dict(required=False, type='str', choices=['lm_ntlm_ntlmv2_krb', 'ntlm_ntlmv2_krb', 'ntlmv2_krb', 'krb']), try_ldap_channel_binding=dict(required=False, type='bool'), use_ldaps=dict(required=False, type='bool'), use_start_tls=dict(required=False, type='bool'), is_multichannel_enabled=dict(required=False, type='bool')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True, mutually_exclusive=[('use_ldaps', 'use_start_tls')])
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.parameters['cifs_server_name'] = self.parameters['name']
        self.rest_api = OntapRestAPI(self.module)
        unsupported_rest_properties = ['workgroup']
        partially_supported_rest_properties = [['encrypt_dc_connection', (9, 8)], ['lm_compatibility_level', (9, 8)], ['aes_netlogon_enabled', (9, 10, 1)], ['ldap_referral_enabled', (9, 10, 1)], ['session_security', (9, 10, 1)], ['try_ldap_channel_binding', (9, 10, 1)], ['use_ldaps', (9, 10, 1)], ['use_start_tls', (9, 10, 1)], ['is_multichannel_enabled', (9, 10, 1)], ['force', (9, 11)], ['default_site', (9, 13, 1)]]
        self.use_rest = self.rest_api.is_rest_supported_properties(self.parameters, unsupported_rest_properties, partially_supported_rest_properties)
        if not self.use_rest:
            unsupported_zapi_properties = ['smb_signing', 'encrypt_dc_connection', 'kdc_encryption', 'smb_encryption', 'restrict_anonymous', 'aes_netlogon_enabled', 'ldap_referral_enabled', 'try_ldap_channel_binding', 'session_security', 'lm_compatibility_level', 'use_ldaps', 'use_start_tls', 'from_name', 'default_site', 'is_multichannel_enabled']
            used_unsupported_zapi_properties = [option for option in unsupported_zapi_properties if option in self.parameters]
            if used_unsupported_zapi_properties:
                self.module.fail_json(msg='Error: %s options supported only with REST.' % ' ,'.join(used_unsupported_zapi_properties))
            if not netapp_utils.has_netapp_lib():
                self.module.fail_json(msg=netapp_utils.netapp_lib_is_required())
            self.server = netapp_utils.setup_na_ontap_zapi(module=self.module, vserver=self.parameters['vserver'])

    def get_cifs_server(self):
        """
        Return details about the CIFS-server
        :param:
            name : Name of the name of the cifs_server

        :return: Details about the cifs_server. None if not found.
        :rtype: dict
        """
        cifs_server_info = netapp_utils.zapi.NaElement('cifs-server-get-iter')
        cifs_server_attributes = netapp_utils.zapi.NaElement('cifs-server-config')
        cifs_server_attributes.add_new_child('cifs-server', self.parameters['cifs_server_name'])
        cifs_server_attributes.add_new_child('vserver', self.parameters['vserver'])
        query = netapp_utils.zapi.NaElement('query')
        query.add_child_elem(cifs_server_attributes)
        cifs_server_info.add_child_elem(query)
        result = self.server.invoke_successfully(cifs_server_info, True)
        return_value = None
        if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
            cifs_server_attributes = result.get_child_by_name('attributes-list').get_child_by_name('cifs-server-config')
            service_state = cifs_server_attributes.get_child_content('administrative-status')
            return_value = {'cifs_server_name': self.parameters['cifs_server_name'], 'service_state': 'started' if service_state == 'up' else 'stopped'}
        return return_value

    def create_cifs_server(self):
        """
        calling zapi to create cifs_server
        """
        options = {'cifs-server': self.parameters['cifs_server_name']}
        if 'service_state' in self.parameters:
            options['administrative-status'] = 'up' if self.parameters['service_state'] == 'started' else 'down'
        if 'workgroup' in self.parameters:
            options['workgroup'] = self.parameters['workgroup']
        if 'domain' in self.parameters:
            options['domain'] = self.parameters['domain']
        if 'admin_user_name' in self.parameters:
            options['admin-username'] = self.parameters['admin_user_name']
        if 'admin_password' in self.parameters:
            options['admin-password'] = self.parameters['admin_password']
        if 'ou' in self.parameters:
            options['organizational-unit'] = self.parameters['ou']
        if 'force' in self.parameters:
            options['force-account-overwrite'] = str(self.parameters['force']).lower()
        cifs_server_create = netapp_utils.zapi.NaElement.create_node_with_children('cifs-server-create', **options)
        try:
            self.server.invoke_successfully(cifs_server_create, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as exc:
            self.module.fail_json(msg='Error Creating cifs_server %s: %s' % (self.parameters['cifs_server_name'], to_native(exc)), exception=traceback.format_exc())

    def delete_cifs_server(self):
        """
        calling zapi to create cifs_server
        """
        options = {}
        if 'admin_user_name' in self.parameters:
            options['admin-username'] = self.parameters['admin_user_name']
        if 'admin_password' in self.parameters:
            options['admin-password'] = self.parameters['admin_password']
        if 'force' in self.parameters:
            options['force-account-delete'] = str(self.parameters['force']).lower()
        if options:
            cifs_server_delete = netapp_utils.zapi.NaElement.create_node_with_children('cifs-server-delete', **options)
        else:
            cifs_server_delete = netapp_utils.zapi.NaElement.create_node_with_children('cifs-server-delete')
        try:
            self.server.invoke_successfully(cifs_server_delete, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as exc:
            self.module.fail_json(msg='Error deleting cifs_server %s: %s' % (self.parameters['cifs_server_name'], to_native(exc)), exception=traceback.format_exc())

    def start_cifs_server(self):
        """
        RModify the cifs_server.
        """
        cifs_server_modify = netapp_utils.zapi.NaElement.create_node_with_children('cifs-server-start')
        try:
            self.server.invoke_successfully(cifs_server_modify, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as e:
            self.module.fail_json(msg='Error modifying cifs_server %s: %s' % (self.parameters['cifs_server_name'], to_native(e)), exception=traceback.format_exc())

    def stop_cifs_server(self):
        """
        RModify the cifs_server.
        """
        cifs_server_modify = netapp_utils.zapi.NaElement.create_node_with_children('cifs-server-stop')
        try:
            self.server.invoke_successfully(cifs_server_modify, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as e:
            self.module.fail_json(msg='Error modifying cifs_server %s: %s' % (self.parameters['cifs_server_name'], to_native(e)), exception=traceback.format_exc())

    def get_cifs_server_rest(self, from_name=None):
        """
        get details of the cifs_server.
        """
        if not self.use_rest:
            return self.get_cifs_server()
        query = {'svm.name': self.parameters['vserver'], 'fields': 'svm.uuid,enabled,security.smb_encryption,security.kdc_encryption,security.smb_signing,security.restrict_anonymous,'}
        query['name'] = from_name or self.parameters['cifs_server_name']
        api = 'protocols/cifs/services'
        if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 8):
            security_option_9_8 = 'security.encrypt_dc_connection,security.lm_compatibility_level,'
            query['fields'] += security_option_9_8
        if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 10, 1):
            security_option_9_10 = 'security.use_ldaps,security.use_start_tls,security.try_ldap_channel_binding,security.session_security,security.ldap_referral_enabled,security.aes_netlogon_enabled,'
            query['fields'] += security_option_9_10
        if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 10, 1):
            service_option_9_10 = 'options.multichannel,'
            query['fields'] += service_option_9_10
        record, error = rest_generic.get_one_record(self.rest_api, api, query)
        if error:
            self.module.fail_json(msg='Error on fetching cifs: %s' % error)
        if record:
            record['service_state'] = 'started' if record.pop('enabled') else 'stopped'
            return {'svm': {'uuid': self.na_helper.safe_get(record, ['svm', 'uuid'])}, 'cifs_server_name': self.na_helper.safe_get(record, ['name']), 'service_state': self.na_helper.safe_get(record, ['service_state']), 'smb_signing': self.na_helper.safe_get(record, ['security', 'smb_signing']), 'encrypt_dc_connection': self.na_helper.safe_get(record, ['security', 'encrypt_dc_connection']), 'kdc_encryption': self.na_helper.safe_get(record, ['security', 'kdc_encryption']), 'smb_encryption': self.na_helper.safe_get(record, ['security', 'smb_encryption']), 'aes_netlogon_enabled': self.na_helper.safe_get(record, ['security', 'aes_netlogon_enabled']), 'ldap_referral_enabled': self.na_helper.safe_get(record, ['security', 'ldap_referral_enabled']), 'session_security': self.na_helper.safe_get(record, ['security', 'session_security']), 'lm_compatibility_level': self.na_helper.safe_get(record, ['security', 'lm_compatibility_level']), 'try_ldap_channel_binding': self.na_helper.safe_get(record, ['security', 'try_ldap_channel_binding']), 'use_ldaps': self.na_helper.safe_get(record, ['security', 'use_ldaps']), 'use_start_tls': self.na_helper.safe_get(record, ['security', 'use_start_tls']), 'restrict_anonymous': self.na_helper.safe_get(record, ['security', 'restrict_anonymous']), 'is_multichannel_enabled': self.na_helper.safe_get(record, ['options', 'multichannel'])}
        return record

    def build_ad_domain(self):
        ad_domain = {}
        if 'admin_user_name' in self.parameters:
            ad_domain['user'] = self.parameters['admin_user_name']
        if 'admin_password' in self.parameters:
            ad_domain['password'] = self.parameters['admin_password']
        if 'ou' in self.parameters:
            ad_domain['organizational_unit'] = self.parameters['ou']
        if 'domain' in self.parameters:
            ad_domain['fqdn'] = self.parameters['domain']
        if 'default_site' in self.parameters:
            ad_domain['default_site'] = self.parameters['default_site']
        return ad_domain

    def create_modify_body_rest(self, params=None):
        """
        Function to define body for create and modify cifs server
        """
        body, query, security, service_options = ({}, {}, {}, {})
        if params is None:
            params = self.parameters
        security_options = ['smb_signing', 'encrypt_dc_connection', 'kdc_encryption', 'smb_encryption', 'restrict_anonymous', 'aes_netlogon_enabled', 'ldap_referral_enabled', 'try_ldap_channel_binding', 'session_security', 'lm_compatibility_level', 'use_ldaps', 'use_start_tls']
        ad_domain = self.build_ad_domain()
        if ad_domain:
            body['ad_domain'] = ad_domain
        if 'force' in self.parameters:
            query['force'] = self.parameters['force']
        for key in security_options:
            if key in params:
                security[key] = params[key]
        if security:
            body['security'] = security
        for key, option in [('multichannel', 'is_multichannel_enabled')]:
            if option in params:
                service_options.update({key: params[option]})
        if service_options:
            body['options'] = service_options
        if 'vserver' in params:
            body['svm.name'] = params['vserver']
        if 'cifs_server_name' in params:
            body['name'] = self.parameters['cifs_server_name']
        if 'service_state' in params:
            body['enabled'] = params['service_state'] == 'started'
        return (body, query)

    def create_cifs_server_rest(self):
        """
        create the cifs_server.
        """
        if not self.use_rest:
            return self.create_cifs_server()
        body, query = self.create_modify_body_rest()
        api = 'protocols/cifs/services'
        dummy, error = rest_generic.post_async(self.rest_api, api, body, query)
        if error is not None:
            self.module.fail_json(msg='Error on creating cifs: %s' % error)

    def delete_cifs_server_rest(self, current):
        """
        delete the cifs_server.
        """
        if not self.use_rest:
            return self.delete_cifs_server()
        ad_domain = self.build_ad_domain()
        body = {'ad_domain': ad_domain} if ad_domain else None
        query = {}
        if 'force' in self.parameters:
            query['force'] = self.parameters['force']
        api = 'protocols/cifs/services'
        dummy, error = rest_generic.delete_async(self.rest_api, api, current['svm']['uuid'], query, body=body)
        if error is not None:
            self.module.fail_json(msg='Error on deleting cifs server: %s' % error)

    def modify_cifs_server_rest(self, current, modify):
        """
        Modify the state of CIFS server.
        rename: cifs server should be in stopped state
        """
        if not self.use_rest:
            return self.modify_cifs_server()
        body, query = self.create_modify_body_rest(modify)
        api = 'protocols/cifs/services'
        dummy, error = rest_generic.patch_async(self.rest_api, api, current['svm']['uuid'], body, query)
        if error is not None:
            self.module.fail_json(msg='Error on modifying cifs server: %s' % error)

    def modify_cifs_server(self):
        """
        Start or stop cifs server in ZAPI.
        """
        if self.parameters.get('service_state') == 'stopped':
            self.stop_cifs_server()
        else:
            self.start_cifs_server()

    def apply(self):
        current = self.get_cifs_server_rest()
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if cd_action == 'create' and 'from_name' in self.parameters:
            current = self.get_cifs_server_rest(self.parameters['from_name'])
            if current is None:
                self.module.fail_json(msg='Error renaming cifs server: %s - no cifs server with from_name: %s.' % (self.parameters['name'], self.parameters['from_name']))
            if not self.parameters.get('force'):
                self.module.fail_json(msg='Error: cannot rename cifs server from %s to %s without force.' % (self.parameters['from_name'], self.parameters['name']))
            cd_action = None
        modify = self.na_helper.get_modified_attributes(current, self.parameters) if cd_action is None else None
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_cifs_server_rest()
            elif cd_action == 'delete':
                self.delete_cifs_server_rest(current)
            else:
                self.modify_cifs_server_rest(current, modify)
                if 'cifs_server_name' in modify and self.parameters.get('service_state') == 'stopped':
                    self.modify_cifs_server_rest(current, {'service_state': 'stopped'})
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify)
        self.module.exit_json(**result)