from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppONTAPVserverAudit:
    """
    Class with vserver audit configuration methods
    """

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), vserver=dict(required=True, type='str'), enabled=dict(required=False, type='bool'), guarantee=dict(required=False, type='bool'), log_path=dict(required=False, type='str'), log=dict(type='dict', options=dict(format=dict(type='str', choices=['xml', 'evtx']), retention=dict(type='dict', options=dict(count=dict(type='int'), duration=dict(type='str'))), rotation=dict(type='dict', options=dict(size=dict(type='int'))))), events=dict(type='dict', options=dict(authorization_policy=dict(type='bool'), cap_staging=dict(type='bool'), cifs_logon_logoff=dict(type='bool'), file_operations=dict(type='bool'), file_share=dict(type='bool'), security_group=dict(type='bool'), user_account=dict(type='bool')))))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.filter_out_none_entries(self.na_helper.set_parameters(self.module.params))
        self.rest_api = netapp_utils.OntapRestAPI(self.module)
        self.rest_api.fail_if_not_rest_minimum_version('na_ontap_vserver_audit', 9, 6)
        partially_supported_rest_properties = [['guarantee', (9, 10, 1)]]
        self.use_rest = self.rest_api.is_rest_supported_properties(self.parameters, None, partially_supported_rest_properties)
        self.svm_uuid = None
        if 'events' in self.parameters and self.parameters['state'] == 'present':
            if all((self.parameters['events'][value] is False for value in self.parameters['events'])) is True:
                self.module.fail_json(msg='Error: At least one event should be enabled')

    def get_vserver_audit_configuration_rest(self):
        """
        Retrieves audit configurations.
        """
        api = 'protocols/audit'
        query = {'svm.name': self.parameters['vserver'], 'fields': 'svm.uuid,enabled,events,log,log_path,'}
        if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 10, 1):
            query['fields'] += 'guarantee,'
        record, error = rest_generic.get_one_record(self.rest_api, api, query)
        if error:
            self.module.fail_json(msg='Error on fetching vserver audit configuration: %s' % error)
        if record:
            self.svm_uuid = self.na_helper.safe_get(record, ['svm', 'uuid'])
            return {'enabled': self.na_helper.safe_get(record, ['enabled']), 'events': self.na_helper.safe_get(record, ['events']), 'log': self.na_helper.safe_get(record, ['log']), 'log_path': self.na_helper.safe_get(record, ['log_path']), 'guarantee': record.get('guarantee', False)}
        return record

    def create_vserver_audit_config_body_rest(self):
        """
        Vserver audit config body for create and modify with rest API.
        """
        body = {}
        if 'events' in self.parameters:
            body['events'] = self.parameters['events']
        if 'guarantee' in self.parameters:
            body['guarantee'] = self.parameters['guarantee']
        if self.na_helper.safe_get(self.parameters, ['log', 'retention', 'count']):
            body['log.retention.count'] = self.parameters['log']['retention']['count']
        if self.na_helper.safe_get(self.parameters, ['log', 'retention', 'duration']):
            body['log.retention.duration'] = self.parameters['log']['retention']['duration']
        if self.na_helper.safe_get(self.parameters, ['log', 'rotation', 'size']):
            body['log.rotation.size'] = self.parameters['log']['rotation']['size']
        if self.na_helper.safe_get(self.parameters, ['log', 'format']):
            body['log.format'] = self.parameters['log']['format']
        if 'log_path' in self.parameters:
            body['log_path'] = self.parameters['log_path']
        return body

    def create_vserver_audit_configuration_rest(self):
        """
        Creates an audit configuration.
        """
        api = 'protocols/audit'
        body = self.create_vserver_audit_config_body_rest()
        if 'vserver' in self.parameters:
            body['svm.name'] = self.parameters.get('vserver')
        if 'enabled' in self.parameters:
            body['enabled'] = self.parameters['enabled']
        record, error = rest_generic.post_async(self.rest_api, api, body)
        if error:
            self.module.fail_json(msg='Error on creating vserver audit configuration: %s' % error)

    def delete_vserver_audit_configuration_rest(self, current):
        """
        Deletes an audit configuration.
        """
        api = 'protocols/audit/%s' % self.svm_uuid
        if current['enabled'] is True:
            modify = {'enabled': False}
            self.modify_vserver_audit_configuration_rest(modify)
            current = self.get_vserver_audit_configuration_rest()
        retry = 2
        while retry > 0:
            record, error = rest_generic.delete_async(self.rest_api, api, None)
            if error and '9699350' in error:
                time.sleep(120)
                retry -= 1
            elif error:
                self.module.fail_json(msg='Error on deleting vserver audit configuration: %s' % error)
            else:
                return

    def modify_vserver_audit_configuration_rest(self, modify):
        """
        Updates audit configuration.
        """
        body = {}
        if 'enabled' in modify:
            body['enabled'] = modify['enabled']
        else:
            body = self.create_vserver_audit_config_body_rest()
        api = 'protocols/audit'
        record, error = rest_generic.patch_async(self.rest_api, api, self.svm_uuid, body)
        if error:
            self.module.fail_json(msg='Error on modifying vserver audit configuration: %s' % error)

    def apply(self):
        current = self.get_vserver_audit_configuration_rest()
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        modify = self.na_helper.get_modified_attributes(current, self.parameters) if cd_action is None else None
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_vserver_audit_configuration_rest()
            elif cd_action == 'delete':
                self.delete_vserver_audit_configuration_rest(current)
            elif modify:
                if 'enabled' in modify:
                    self.modify_vserver_audit_configuration_rest(modify)
                    modify.pop('enabled')
                if modify:
                    self.modify_vserver_audit_configuration_rest(modify)
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action)
        self.module.exit_json(**result)