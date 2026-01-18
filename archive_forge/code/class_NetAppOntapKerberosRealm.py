from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppOntapKerberosRealm:
    """
    Kerberos Realm definition class
    """

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(admin_server_ip=dict(required=False, type='str'), admin_server_port=dict(required=False, type='str'), clock_skew=dict(required=False, type='str'), comment=dict(required=False, type='str'), kdc_ip=dict(required=False, type='str'), kdc_port=dict(required=False, type='int'), kdc_vendor=dict(required=False, type='str', choices=['microsoft', 'other']), pw_server_ip=dict(required=False, type='str'), pw_server_port=dict(required=False, type='str'), realm=dict(required=True, type='str'), state=dict(required=False, choices=['present', 'absent'], default='present'), vserver=dict(required=True, type='str'), ad_server_ip=dict(required=False, type='str'), ad_server_name=dict(required=False, type='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True, required_if=[('state', 'present', ['kdc_vendor', 'kdc_ip']), ('kdc_vendor', 'microsoft', ['ad_server_ip', 'ad_server_name'])])
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = netapp_utils.OntapRestAPI(self.module)
        partially_supported_rest_properties = [['admin_server_ip', (9, 13, 1)], ['admin_server_port', (9, 13, 1)], ['clock_skew', (9, 13, 1)], ['pw_server_ip', (9, 13, 1)], ['pw_server_port', (9, 13, 1)]]
        self.use_rest = self.rest_api.is_rest_supported_properties(self.parameters, None, partially_supported_rest_properties)
        self.svm_uuid = None
        if not self.use_rest:
            if not netapp_utils.has_netapp_lib():
                self.module.fail_json(msg=netapp_utils.netapp_lib_is_required())
            self.server = netapp_utils.setup_na_ontap_zapi(module=self.module, vserver=self.parameters['vserver'])
            self.simple_attributes = ['admin_server_ip', 'admin_server_port', 'clock_skew', 'kdc_ip', 'kdc_vendor']

    def get_krbrealm(self):
        """
        Checks if Kerberos Realm config exists.

        :return:
            kerberos realm object if found
            None if not found
        :rtype: object/None
        """
        if self.use_rest:
            return self.get_krbrealm_rest()
        krbrealm_info = netapp_utils.zapi.NaElement('kerberos-realm-get-iter')
        query_details = netapp_utils.zapi.NaElement.create_node_with_children('kerberos-realm', **{'realm': self.parameters['realm'], 'vserver-name': self.parameters['vserver']})
        query = netapp_utils.zapi.NaElement('query')
        query.add_child_elem(query_details)
        krbrealm_info.add_child_elem(query)
        try:
            result = self.server.invoke_successfully(krbrealm_info, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error fetching kerberos realm %s: %s' % (self.parameters['realm'], to_native(error)))
        krbrealm_details = None
        if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
            attributes_list = result.get_child_by_name('attributes-list')
            config_info = attributes_list.get_child_by_name('kerberos-realm')
            krbrealm_details = {'admin_server_ip': config_info.get_child_content('admin-server-ip'), 'admin_server_port': config_info.get_child_content('admin-server-port'), 'clock_skew': config_info.get_child_content('clock-skew'), 'kdc_ip': config_info.get_child_content('kdc-ip'), 'kdc_port': int(config_info.get_child_content('kdc-port')), 'kdc_vendor': config_info.get_child_content('kdc-vendor'), 'pw_server_ip': config_info.get_child_content('password-server-ip'), 'pw_server_port': config_info.get_child_content('password-server-port'), 'realm': config_info.get_child_content('realm'), 'vserver': config_info.get_child_content('vserver-name'), 'ad_server_ip': config_info.get_child_content('ad-server-ip'), 'ad_server_name': config_info.get_child_content('ad-server-name'), 'comment': config_info.get_child_content('comment')}
        return krbrealm_details

    def create_krbrealm(self):
        """supported
        Create Kerberos Realm configuration
        """
        if self.use_rest:
            return self.create_krbrealm_rest()
        options = {'realm': self.parameters['realm']}
        for attribute in self.simple_attributes:
            if self.parameters.get(attribute) is not None:
                options[str(attribute).replace('_', '-')] = self.parameters[attribute]
        if self.parameters.get('kdc_port'):
            options['kdc-port'] = str(self.parameters['kdc_port'])
        if self.parameters.get('pw_server_ip') is not None:
            options['password-server-ip'] = self.parameters['pw_server_ip']
        if self.parameters.get('pw_server_port') is not None:
            options['password-server-port'] = self.parameters['pw_server_port']
        if self.parameters.get('ad_server_ip') is not None:
            options['ad-server-ip'] = self.parameters['ad_server_ip']
        if self.parameters.get('ad_server_name') is not None:
            options['ad-server-name'] = self.parameters['ad_server_name']
        krbrealm_create = netapp_utils.zapi.NaElement.create_node_with_children('kerberos-realm-create', **options)
        try:
            self.server.invoke_successfully(krbrealm_create, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as errcatch:
            self.module.fail_json(msg='Error creating Kerberos Realm configuration %s: %s' % (self.parameters['realm'], to_native(errcatch)), exception=traceback.format_exc())

    def delete_krbrealm(self):
        """
        Delete Kerberos Realm configuration
        """
        if self.use_rest:
            return self.delete_krbrealm_rest()
        krbrealm_delete = netapp_utils.zapi.NaElement.create_node_with_children('kerberos-realm-delete', **{'realm': self.parameters['realm']})
        try:
            self.server.invoke_successfully(krbrealm_delete, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as errcatch:
            self.module.fail_json(msg='Error deleting Kerberos Realm configuration %s: %s' % (self.parameters['realm'], to_native(errcatch)), exception=traceback.format_exc())

    def modify_krbrealm(self, modify):
        """
        Modify Kerberos Realm
        :param modify: list of modify attributes
        """
        if self.use_rest:
            return self.modify_krbrealm_rest(modify)
        krbrealm_modify = netapp_utils.zapi.NaElement('kerberos-realm-modify')
        krbrealm_modify.add_new_child('realm', self.parameters['realm'])
        for attribute in modify:
            if attribute in self.simple_attributes:
                krbrealm_modify.add_new_child(str(attribute).replace('_', '-'), self.parameters[attribute])
            if attribute == 'kdc_port':
                krbrealm_modify.add_new_child('kdc-port', str(self.parameters['kdc_port']))
            if attribute == 'pw_server_ip':
                krbrealm_modify.add_new_child('password-server-ip', self.parameters['pw_server_ip'])
            if attribute == 'pw_server_port':
                krbrealm_modify.add_new_child('password-server-port', self.parameters['pw_server_port'])
            if attribute == 'ad_server_ip':
                krbrealm_modify.add_new_child('ad-server-ip', self.parameters['ad_server_ip'])
            if attribute == 'ad_server_name':
                krbrealm_modify.add_new_child('ad-server-name', self.parameters['ad_server_name'])
            if attribute == 'comment':
                krbrealm_modify.add_new_child('comment', self.parameters['comment'])
        try:
            self.server.invoke_successfully(krbrealm_modify, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as errcatch:
            self.module.fail_json(msg='Error modifying Kerberos Realm %s: %s' % (self.parameters['realm'], to_native(errcatch)), exception=traceback.format_exc())

    def get_krbrealm_rest(self):
        api = 'protocols/nfs/kerberos/realms'
        params = {'name': self.parameters['realm'], 'svm.name': self.parameters['vserver'], 'fields': 'kdc,ad_server,svm,comment,password_server,admin_server,clock_skew'}
        record, error = rest_generic.get_one_record(self.rest_api, api, params)
        if error:
            self.module.fail_json(msg='Error fetching kerberos realm %s: %s' % (self.parameters['realm'], to_native(error)))
        if record:
            self.svm_uuid = record['svm']['uuid']
            return {'kdc_ip': self.na_helper.safe_get(record, ['kdc', 'ip']), 'kdc_port': self.na_helper.safe_get(record, ['kdc', 'port']), 'kdc_vendor': self.na_helper.safe_get(record, ['kdc', 'vendor']), 'ad_server_ip': self.na_helper.safe_get(record, ['ad_server', 'address']), 'ad_server_name': self.na_helper.safe_get(record, ['ad_server', 'name']), 'comment': self.na_helper.safe_get(record, ['comment']), 'pw_server_ip': self.na_helper.safe_get(record, ['password_server', 'address']), 'pw_server_port': str(self.na_helper.safe_get(record, ['password_server', 'port'])), 'admin_server_ip': self.na_helper.safe_get(record, ['admin_server', 'address']), 'admin_server_port': str(self.na_helper.safe_get(record, ['admin_server', 'port'])), 'clock_skew': str(self.na_helper.safe_get(record, ['clock_skew']))}
        return None

    def create_krbrealm_rest(self):
        api = 'protocols/nfs/kerberos/realms'
        body = {'name': self.parameters['realm'], 'svm.name': self.parameters['vserver'], 'kdc.ip': self.parameters['kdc_ip'], 'kdc.vendor': self.parameters['kdc_vendor']}
        if self.parameters.get('kdc_port'):
            body['kdc.port'] = self.parameters['kdc_port']
        if self.parameters.get('comment'):
            body['comment'] = self.parameters['comment']
        if self.parameters.get('ad_server_ip'):
            body['ad_server.address'] = self.parameters['ad_server_ip']
        if self.parameters.get('ad_server_name'):
            body['ad_server.name'] = self.parameters['ad_server_name']
        if self.parameters.get('admin_server_port'):
            body['admin_server.port'] = self.parameters['admin_server_port']
        if self.parameters.get('pw_server_port'):
            body['password_server.port'] = self.parameters['pw_server_port']
        if self.parameters.get('clock_skew'):
            body['clock_skew'] = self.parameters['clock_skew']
        if self.parameters.get('admin_server_ip'):
            body['admin_server.address'] = self.parameters['admin_server_ip']
        if self.parameters.get('pw_server_ip'):
            body['password_server.address'] = self.parameters['pw_server_ip']
        dummy, error = rest_generic.post_async(self.rest_api, api, body)
        if error:
            self.module.fail_json(msg='Error creating Kerberos Realm configuration %s: %s' % (self.parameters['realm'], to_native(error)))

    def modify_krbrealm_rest(self, modify):
        api = 'protocols/nfs/kerberos/realms/%s' % self.svm_uuid
        body = {}
        if modify.get('kdc_ip'):
            body['kdc.ip'] = modify['kdc_ip']
        if modify.get('kdc_vendor'):
            body['kdc.vendor'] = modify['kdc_vendor']
        if modify.get('kdc_port'):
            body['kdc.port'] = modify['kdc_port']
        if modify.get('comment'):
            body['comment'] = modify['comment']
        if modify.get('ad_server_ip'):
            body['ad_server.address'] = modify['ad_server_ip']
        if modify.get('ad_server_name'):
            body['ad_server.name'] = modify['ad_server_name']
        if modify.get('admin_server_ip'):
            body['admin_server.address'] = modify['admin_server_ip']
        if modify.get('admin_server_port'):
            body['admin_server.port'] = modify['admin_server_port']
        if modify.get('pw_server_ip'):
            body['password_server.address'] = modify['pw_server_ip']
        if modify.get('pw_server_port'):
            body['password_server.port'] = modify['pw_server_port']
        if modify.get('clock_skew'):
            body['clock_skew'] = modify['clock_skew']
        dummy, error = rest_generic.patch_async(self.rest_api, api, self.parameters['realm'], body)
        if error:
            self.module.fail_json(msg='Error modifying Kerberos Realm %s: %s' % (self.parameters['realm'], to_native(error)))

    def delete_krbrealm_rest(self):
        api = 'protocols/nfs/kerberos/realms/%s' % self.svm_uuid
        dummy, error = rest_generic.delete_async(self.rest_api, api, self.parameters['realm'])
        if error:
            self.module.fail_json(msg='Error deleting Kerberos Realm configuration %s: %s' % (self.parameters['realm'], to_native(error)))

    def apply(self):
        """Call create/modify/delete operations."""
        current = self.get_krbrealm()
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        modify = self.na_helper.get_modified_attributes(current, self.parameters)
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_krbrealm()
            elif cd_action == 'delete':
                self.delete_krbrealm()
            elif modify:
                self.modify_krbrealm(modify)
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify)
        self.module.exit_json(**result)