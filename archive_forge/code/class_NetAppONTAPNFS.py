from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppONTAPNFS:
    """ object initialize and class methods """

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), service_state=dict(required=False, type='str', choices=['started', 'stopped']), vserver=dict(required=True, type='str'), nfsv3=dict(required=False, type='str', default=None, choices=['enabled', 'disabled']), nfsv3_fsid_change=dict(required=False, type='str', default=None, choices=['enabled', 'disabled']), nfsv4_fsid_change=dict(required=False, type='str', default=None, choices=['enabled', 'disabled']), nfsv4=dict(required=False, type='str', default=None, choices=['enabled', 'disabled']), nfsv41=dict(required=False, type='str', default=None, choices=['enabled', 'disabled'], aliases=['nfsv4.1']), nfsv41_pnfs=dict(required=False, type='str', default=None, choices=['enabled', 'disabled']), nfsv4_numeric_ids=dict(required=False, type='str', default=None, choices=['enabled', 'disabled']), vstorage_state=dict(required=False, type='str', default=None, choices=['enabled', 'disabled']), tcp=dict(required=False, default=None, type='str', choices=['enabled', 'disabled']), udp=dict(required=False, default=None, type='str', choices=['enabled', 'disabled']), nfsv4_id_domain=dict(required=False, type='str', default=None), nfsv40_acl=dict(required=False, type='str', default=None, choices=['enabled', 'disabled']), nfsv40_read_delegation=dict(required=False, type='str', default=None, choices=['enabled', 'disabled']), nfsv40_referrals=dict(required=False, type='str', default=None, choices=['enabled', 'disabled']), nfsv40_write_delegation=dict(required=False, type='str', default=None, choices=['enabled', 'disabled']), nfsv41_acl=dict(required=False, type='str', default=None, choices=['enabled', 'disabled']), nfsv41_read_delegation=dict(required=False, type='str', default=None, choices=['enabled', 'disabled']), nfsv41_referrals=dict(required=False, type='str', default=None, choices=['enabled', 'disabled']), nfsv41_write_delegation=dict(required=False, type='str', default=None, choices=['enabled', 'disabled']), showmount=dict(required=False, default=None, type='str', choices=['enabled', 'disabled']), tcp_max_xfer_size=dict(required=False, default=None, type='int'), security=dict(type='dict', options=dict(rpcsec_context_idle=dict(required=False, type='int'), ntfs_unix_security=dict(required=False, type='str', choices=['ignore', 'fail', 'use_export_policy']), chown_mode=dict(required=False, type='str', choices=['restricted', 'unrestricted', 'use_export_policy']), nt_acl_display_permission=dict(required=False, type='bool'), permitted_encryption_types=dict(type='list', elements='str', required=False))), root=dict(type='dict', options=dict(ignore_nt_acl=dict(required=False, type='bool'), skip_write_permission_check=dict(required=False, type='bool'))), windows=dict(type='dict', options=dict(map_unknown_uid_to_default_user=dict(required=False, type='bool'), v3_ms_dos_client_enabled=dict(required=False, type='bool'), default_user=dict(required=False, type='str')))))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.zapi_names = {'nfsv3': 'is-nfsv3-enabled', 'nfsv3_fsid_change': 'is-nfsv3-fsid-change-enabled', 'nfsv4_fsid_change': 'is-nfsv4-fsid-change-enabled', 'nfsv4': 'is-nfsv40-enabled', 'nfsv41': 'is-nfsv41-enabled', 'nfsv41_pnfs': 'is-nfsv41-pnfs-enabled', 'nfsv4_numeric_ids': 'is-nfsv4-numeric-ids-enabled', 'vstorage_state': 'is-vstorage-enabled', 'nfsv4_id_domain': 'nfsv4-id-domain', 'tcp': 'is-tcp-enabled', 'udp': 'is-udp-enabled', 'nfsv40_acl': 'is-nfsv40-acl-enabled', 'nfsv40_read_delegation': 'is-nfsv40-read-delegation-enabled', 'nfsv40_referrals': 'is-nfsv40-referrals-enabled', 'nfsv40_write_delegation': 'is-nfsv40-write-delegation-enabled', 'nfsv41_acl': 'is-nfsv41-acl-enabled', 'nfsv41_read_delegation': 'is-nfsv41-read-delegation-enabled', 'nfsv41_referrals': 'is-nfsv41-referrals-enabled', 'nfsv41_write_delegation': 'is-nfsv41-write-delegation-enabled', 'showmount': 'showmount', 'tcp_max_xfer_size': 'tcp-max-xfer-size'}
        self.rest_api = netapp_utils.OntapRestAPI(self.module)
        unsupported_rest_properties = ['nfsv3_fsid_change', 'nfsv4_fsid_change', 'nfsv4_numeric_ids', 'nfsv40_referrals', 'nfsv41_referrals']
        partially_supported_rest_properties = [['showmount', (9, 8)], ['root', (9, 11, 0)], ['windows', (9, 11, 0)], ['security', (9, 11, 0)], ['tcp_max_xfer_size', (9, 11, 0)]]
        self.use_rest = self.rest_api.is_rest_supported_properties(self.parameters, unsupported_rest_properties, partially_supported_rest_properties)
        if 'nfsv4.1' in self.parameters:
            self.module.warn('Error: "nfsv4.1" option conflicts with Ansible naming conventions - please use "nfsv41".')
        self.svm_uuid = None
        self.unsupported_zapi_properties = ['root', 'windows', 'security']
        self.parameters = self.na_helper.filter_out_none_entries(self.parameters)
        if not self.use_rest:
            if not netapp_utils.has_netapp_lib():
                self.module.fail_json(msg=netapp_utils.netapp_lib_is_required())
            for unsupported_zapi_property in self.unsupported_zapi_properties:
                if self.parameters.get(unsupported_zapi_property) is not None:
                    msg = 'Error: %s option is not supported with ZAPI.  It can only be used with REST.' % unsupported_zapi_property
                    self.module.fail_json(msg=msg)
            self.server = netapp_utils.setup_na_ontap_zapi(module=self.module, vserver=self.parameters['vserver'])

    def get_nfs_service(self):
        if self.use_rest:
            return self.get_nfs_service_rest()
        nfs_get_iter = netapp_utils.zapi.NaElement('nfs-service-get-iter')
        nfs_info = netapp_utils.zapi.NaElement('nfs-info')
        nfs_info.add_new_child('vserver', self.parameters['vserver'])
        query = netapp_utils.zapi.NaElement('query')
        query.add_child_elem(nfs_info)
        nfs_get_iter.add_child_elem(query)
        result = self.server.invoke_successfully(nfs_get_iter, True)
        if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) >= 1:
            return self.format_return(result)
        return None

    def format_return(self, result):
        attributes_list = result.get_child_by_name('attributes-list').get_child_by_name('nfs-info')
        return {'nfsv3': self.convert_from_bool(attributes_list.get_child_content('is-nfsv3-enabled')), 'nfsv3_fsid_change': self.convert_from_bool(attributes_list.get_child_content('is-nfsv3-fsid-change-enabled')), 'nfsv4_fsid_change': self.convert_from_bool(attributes_list.get_child_content('is-nfsv4-fsid-change-enabled')), 'nfsv4': self.convert_from_bool(attributes_list.get_child_content('is-nfsv40-enabled')), 'nfsv41': self.convert_from_bool(attributes_list.get_child_content('is-nfsv41-enabled')), 'nfsv41_pnfs': self.convert_from_bool(attributes_list.get_child_content('is-nfsv41-pnfs-enabled')), 'nfsv4_numeric_ids': self.convert_from_bool(attributes_list.get_child_content('is-nfsv4-numeric-ids-enabled')), 'vstorage_state': self.convert_from_bool(attributes_list.get_child_content('is-vstorage-enabled')), 'nfsv4_id_domain': attributes_list.get_child_content('nfsv4-id-domain'), 'tcp': self.convert_from_bool(attributes_list.get_child_content('is-tcp-enabled')), 'udp': self.convert_from_bool(attributes_list.get_child_content('is-udp-enabled')), 'nfsv40_acl': self.convert_from_bool(attributes_list.get_child_content('is-nfsv40-acl-enabled')), 'nfsv40_read_delegation': self.convert_from_bool(attributes_list.get_child_content('is-nfsv40-read-delegation-enabled')), 'nfsv40_referrals': self.convert_from_bool(attributes_list.get_child_content('is-nfsv40-referrals-enabled')), 'nfsv40_write_delegation': self.convert_from_bool(attributes_list.get_child_content('is-nfsv40-write-delegation-enabled')), 'nfsv41_acl': self.convert_from_bool(attributes_list.get_child_content('is-nfsv41-acl-enabled')), 'nfsv41_read_delegation': self.convert_from_bool(attributes_list.get_child_content('is-nfsv41-read-delegation-enabled')), 'nfsv41_referrals': self.convert_from_bool(attributes_list.get_child_content('is-nfsv41-referrals-enabled')), 'nfsv41_write_delegation': self.convert_from_bool(attributes_list.get_child_content('is-nfsv41-write-delegation-enabled')), 'showmount': self.convert_from_bool(attributes_list.get_child_content('showmount')), 'tcp_max_xfer_size': self.na_helper.get_value_for_int(True, attributes_list.get_child_content('tcp-max-xfer-size'))}

    def get_nfs_status(self):
        nfs_status = netapp_utils.zapi.NaElement('nfs-status')
        result = self.server.invoke_successfully(nfs_status, True)
        return result.get_child_content('is-enabled')

    def create_nfs_service(self):
        if self.use_rest:
            return self.create_nfs_service_rest()
        self.enable_nfs()

    def enable_nfs(self):
        """
        enable nfs (online). If the NFS service was not explicitly created,
        this API will create one with default options.
        """
        nfs_enable = netapp_utils.zapi.NaElement.create_node_with_children('nfs-enable')
        try:
            self.server.invoke_successfully(nfs_enable, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error changing the service_state of nfs %s to %s: %s' % (self.parameters['vserver'], self.parameters['service_state'], to_native(error)), exception=traceback.format_exc())

    def disable_nfs(self):
        """
        disable nfs (offline).
        """
        nfs_disable = netapp_utils.zapi.NaElement.create_node_with_children('nfs-disable')
        try:
            self.server.invoke_successfully(nfs_disable, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error changing the service_state of nfs %s to %s: %s' % (self.parameters['vserver'], self.parameters['service_state'], to_native(error)), exception=traceback.format_exc())

    def modify_nfs_service(self, modify):
        if self.use_rest:
            return self.modify_nfs_service_rest(modify)
        nfs_modify = netapp_utils.zapi.NaElement('nfs-service-modify')
        service_state = modify.pop('service_state', None)
        self.modify_service_state(service_state)
        for each in modify:
            if each in ['nfsv4_id_domain', 'tcp_max_xfer_size']:
                nfs_modify.add_new_child(self.zapi_names[each], str(modify[each]))
            else:
                nfs_modify.add_new_child(self.zapi_names[each], self.convert_to_bool(modify[each]))
        try:
            self.server.invoke_successfully(nfs_modify, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error modifying nfs: %s' % to_native(error), exception=traceback.format_exc())

    def modify_service_state(self, service_state):
        nfs_enabled = self.get_nfs_status()
        if service_state == 'started' and nfs_enabled == 'false':
            self.enable_nfs()
        elif service_state == 'stopped' and nfs_enabled == 'true':
            self.disable_nfs()

    def delete_nfs_service(self):
        """
        delete nfs service.
        """
        if self.use_rest:
            return self.delete_nfs_service_rest()
        nfs_delete = netapp_utils.zapi.NaElement.create_node_with_children('nfs-service-destroy')
        try:
            self.server.invoke_successfully(nfs_delete, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error deleting nfs: %s' % to_native(error), exception=traceback.format_exc())

    def get_nfs_service_rest(self):
        api = 'protocols/nfs/services'
        params = {'svm.name': self.parameters['vserver'], 'fields': 'protocol.v3_enabled,protocol.v40_enabled,protocol.v41_enabled,protocol.v41_features.pnfs_enabled,vstorage_enabled,protocol.v4_id_domain,transport.tcp_enabled,transport.udp_enabled,protocol.v40_features.acl_enabled,protocol.v40_features.read_delegation_enabled,protocol.v40_features.write_delegation_enabled,protocol.v41_features.acl_enabled,protocol.v41_features.read_delegation_enabled,protocol.v41_features.write_delegation_enabled,enabled,svm.uuid,'}
        if self.parameters.get('showmount'):
            params['fields'] += 'showmount_enabled,'
        if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 11, 0):
            params['fields'] += 'root.*,security.*,windows.*,transport.tcp_max_transfer_size'
        record, error = rest_generic.get_one_record(self.rest_api, api, params)
        if error:
            self.module.fail_json(msg='Error getting nfs services for SVM %s: %s' % (self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())
        if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 11, 0):
            if record and 'default_user' not in record.get('windows'):
                record['windows']['default_user'] = None
        return self.format_get_nfs_service_rest(record) if record else record

    def format_get_nfs_service_rest(self, record):
        return {'nfsv3': self.convert_from_bool(self.na_helper.safe_get(record, ['protocol', 'v3_enabled'])), 'nfsv4': self.convert_from_bool(self.na_helper.safe_get(record, ['protocol', 'v40_enabled'])), 'nfsv41': self.convert_from_bool(self.na_helper.safe_get(record, ['protocol', 'v41_enabled'])), 'nfsv41_pnfs': self.convert_from_bool(self.na_helper.safe_get(record, ['protocol', 'v41_features', 'pnfs_enabled'])), 'vstorage_state': self.convert_from_bool(self.na_helper.safe_get(record, ['vstorage_enabled'])), 'nfsv4_id_domain': self.na_helper.safe_get(record, ['protocol', 'v4_id_domain']), 'tcp': self.convert_from_bool(self.na_helper.safe_get(record, ['transport', 'tcp_enabled'])), 'udp': self.convert_from_bool(self.na_helper.safe_get(record, ['transport', 'udp_enabled'])), 'tcp_max_xfer_size': self.na_helper.safe_get(record, ['transport', 'tcp_max_transfer_size']), 'nfsv40_acl': self.convert_from_bool(self.na_helper.safe_get(record, ['protocol', 'v40_features', 'acl_enabled'])), 'nfsv40_read_delegation': self.convert_from_bool(self.na_helper.safe_get(record, ['protocol', 'v40_features', 'read_delegation_enabled'])), 'nfsv40_write_delegation': self.convert_from_bool(self.na_helper.safe_get(record, ['protocol', 'v40_features', 'write_delegation_enabled'])), 'nfsv41_acl': self.convert_from_bool(self.na_helper.safe_get(record, ['protocol', 'v41_features', 'acl_enabled'])), 'nfsv41_read_delegation': self.convert_from_bool(self.na_helper.safe_get(record, ['protocol', 'v41_features', 'read_delegation_enabled'])), 'nfsv41_write_delegation': self.convert_from_bool(self.na_helper.safe_get(record, ['protocol', 'v41_features', 'write_delegation_enabled'])), 'showmount': self.convert_from_bool(self.na_helper.safe_get(record, ['showmount_enabled'])), 'svm_uuid': self.na_helper.safe_get(record, ['svm', 'uuid']), 'service_state': self.convert_from_bool_to_started(self.na_helper.safe_get(record, ['enabled'])), 'root': self.na_helper.safe_get(record, ['root']), 'windows': self.na_helper.safe_get(record, ['windows']), 'security': self.na_helper.safe_get(record, ['security'])}

    def create_nfs_service_rest(self):
        api = 'protocols/nfs/services'
        body = {'svm.name': self.parameters['vserver']}
        body.update(self.create_modify_body(body))
        dummy, error = rest_generic.post_async(self.rest_api, api, body, job_timeout=120)
        if error:
            self.module.fail_json(msg='Error creating nfs service for SVM %s: %s' % (self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())

    def delete_nfs_service_rest(self):
        if self.svm_uuid is None:
            self.module.fail_json(msg='Error deleting nfs service for SVM %s: svm.uuid is None' % self.parameters['vserver'])
        dummy, error = rest_generic.delete_async(self.rest_api, 'protocols/nfs/services', self.svm_uuid, job_timeout=120)
        if error:
            self.module.fail_json(msg='Error deleting nfs service for SVM %s' % self.parameters['vserver'])

    def modify_nfs_service_rest(self, modify):
        if self.svm_uuid is None:
            self.module.fail_json(msg='Error modifying nfs service for SVM %s: svm.uuid is None' % self.parameters['vserver'])
        api = 'protocols/nfs/services'
        body = {}
        body.update(self.create_modify_body(body, modify))
        dummy, error = rest_generic.patch_async(self.rest_api, api, self.svm_uuid, body, job_timeout=120)
        if error:
            self.module.fail_json(msg='Error modifying nfs service for SVM %s: %s' % (self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())

    def create_modify_body(self, body, modify=None):
        params = modify or self.parameters
        if params.get('nfsv3') is not None:
            body['protocol.v3_enabled'] = self.convert_to_bool(params['nfsv3'])
        if params.get('nfsv4') is not None:
            body['protocol.v40_enabled'] = self.convert_to_bool(params['nfsv4'])
        if params.get('nfsv41') is not None:
            body['protocol.v41_enabled'] = self.convert_to_bool(params['nfsv41'])
        if params.get('nfsv41_pnfs') is not None:
            body['protocol.v41_features.pnfs_enabled'] = self.convert_to_bool(params['nfsv41_pnfs'])
        if params.get('vstorage_state') is not None:
            body['vstorage_enabled'] = self.convert_to_bool(params['vstorage_state'])
        if params.get('nfsv4_id_domain') is not None:
            body['protocol.v4_id_domain'] = params['nfsv4_id_domain']
        if params.get('tcp') is not None:
            body['transport.tcp_enabled'] = self.convert_to_bool(params['tcp'])
        if params.get('udp') is not None:
            body['transport.udp_enabled'] = self.convert_to_bool(params['udp'])
        if params.get('nfsv40_acl') is not None:
            body['protocol.v40_features.acl_enabled'] = self.convert_to_bool(params['nfsv40_acl'])
        if params.get('nfsv40_read_delegation') is not None:
            body['protocol.v40_features.read_delegation_enabled'] = self.convert_to_bool(params['nfsv40_read_delegation'])
        if params.get('nfsv40_write_delegation') is not None:
            body['protocol.v40_features.write_delegation_enabled'] = self.convert_to_bool(params['nfsv40_write_delegation'])
        if params.get('nfsv41_acl') is not None:
            body['protocol.v41_features.acl_enabled'] = self.convert_to_bool(params['nfsv41_acl'])
        if params.get('nfsv41_read_delegation') is not None:
            body['protocol.v41_features.read_delegation_enabled'] = self.convert_to_bool(params['nfsv41_read_delegation'])
        if params.get('nfsv41_write_delegation') is not None:
            body['protocol.v41_features.write_delegation_enabled'] = self.convert_to_bool(params['nfsv41_write_delegation'])
        if params.get('showmount') is not None:
            body['showmount_enabled'] = self.convert_to_bool(params['showmount'])
        if params.get('service_state') is not None:
            body['enabled'] = self.convert_to_bool(params['service_state'])
        if params.get('root') is not None:
            body['root'] = params['root']
        if params.get('windows') is not None:
            body['windows'] = params['windows']
        if params.get('security') is not None:
            body['security'] = params['security']
        if params.get('tcp_max_xfer_size') is not None:
            body['transport.tcp_max_transfer_size'] = params['tcp_max_xfer_size']
        return body

    def convert_to_bool(self, value):
        return 'true' if value in ['enabled', 'started'] else 'false'

    def convert_from_bool(self, value):
        return 'enabled' if value in ['true', True] else 'disabled'

    def convert_from_bool_to_started(self, value):
        return 'started' if value in ['true', True] else 'stopped'

    def validate_modify(self, current, modify):
        """Earlier ONTAP versions do not support tcp_max_xfer_size"""
        if 'tcp_max_xfer_size' in modify and current['tcp_max_xfer_size'] is None:
            self.module.fail_json(msg='Error: tcp_max_xfer_size is not supported on ONTAP 9.3 or earlier.')

    def apply(self):
        current = self.get_nfs_service()
        if self.use_rest and current is not None:
            self.svm_uuid = current.get('svm_uuid')
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        modify = None
        if cd_action is None and self.parameters['state'] == 'present':
            modify = self.na_helper.get_modified_attributes(current, self.parameters)
            if not self.use_rest:
                self.validate_modify(current, modify)
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_nfs_service()
            elif cd_action == 'delete':
                self.delete_nfs_service()
            elif modify:
                self.modify_nfs_service(modify)
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify)
        self.module.exit_json(**result)