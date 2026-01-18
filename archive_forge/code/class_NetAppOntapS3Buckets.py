from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppOntapS3Buckets:

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), name=dict(required=True, type='str'), vserver=dict(required=True, type='str'), aggregates=dict(required=False, type='list', elements='str'), constituents_per_aggregate=dict(required=False, type='int'), size=dict(required=False, type='int'), comment=dict(required=False, type='str'), type=dict(required=False, type='str', choices=['s3', 'nas']), nas_path=dict(required=False, type='str'), policy=dict(type='dict', options=dict(statements=dict(type='list', elements='dict', options=dict(sid=dict(required=False, type='str'), resources=dict(required=False, type='list', elements='str'), actions=dict(required=False, type='list', elements='str'), effect=dict(required=False, type='str', choices=['allow', 'deny']), conditions=dict(type='list', elements='dict', options=dict(operator=dict(required=False, type='str', choices=['ip_address', 'not_ip_address', 'string_equals', 'string_not_equals', 'string_equals_ignore_case', 'string_not_equals_ignore_case', 'string_like', 'string_not_like', 'numeric_equals', 'numeric_not_equals', 'numeric_greater_than', 'numeric_greater_than_equals', 'numeric_less_than', 'numeric_less_than_equals']), max_keys=dict(required=False, type='list', elements='str', no_log=False), delimiters=dict(required=False, type='list', elements='str'), source_ips=dict(required=False, type='list', elements='str'), prefixes=dict(required=False, type='list', elements='str'), usernames=dict(required=False, type='list', elements='str'))), principals=dict(type='list', elements='str'))))), qos_policy=dict(type='dict', options=dict(max_throughput_iops=dict(type='int'), max_throughput_mbps=dict(type='int'), name=dict(type='str'), min_throughput_iops=dict(type='int'), min_throughput_mbps=dict(type='int'))), audit_event_selector=dict(type='dict', options=dict(access=dict(type='str', choices=['read', 'write', 'all']), permission=dict(type='str', choices=['allow', 'deny', 'all'])))))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.svm_uuid = None
        self.uuid = None
        self.volume_uuid = None
        self.na_helper = NetAppModule(self.module)
        self.parameters = self.na_helper.check_and_set_parameters(self.module)
        self.rest_api = netapp_utils.OntapRestAPI(self.module)
        self.rest_api.fail_if_not_rest_minimum_version('na_ontap_s3_bucket', 9, 8)
        partially_supported_rest_properties = [['audit_event_selector', (9, 10, 1)], ['type', (9, 12, 1)], ['nas_path', (9, 12, 1)]]
        self.use_rest = self.rest_api.is_rest_supported_properties(self.parameters, None, partially_supported_rest_properties)
        if self.parameters.get('policy'):
            self.parameters['policy'] = self.na_helper.filter_out_none_entries(self.parameters['policy'], True)
            for statement in self.parameters['policy'].get('statements', []):
                if {} in self.parameters['policy']['statements']:
                    self.module.fail_json(msg='Error: cannot set empty dict for policy statements.')
                if len(statement.get('resources', [])) == 1 and statement['resources'] == ['*']:
                    statement['resources'] = [self.parameters['name'], self.parameters['name'] + '/*']
                for condition in statement.get('conditions', []):
                    updated_ips = []
                    for ip in condition.get('source_ips', []):
                        if '/' in ip:
                            updated_ips.append(ip)
                        else:
                            updated_ips.append(ip + '/32')
                    if updated_ips:
                        condition['source_ips'] = updated_ips

    def get_s3_bucket(self):
        api = 'protocols/s3/buckets'
        fields = 'name,svm.name,size,comment,volume.uuid,policy,policy.statements,qos_policy'
        if self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 12, 1):
            fields += ',audit_event_selector,type,nas_path'
        elif self.rest_api.meets_rest_minimum_version(self.use_rest, 9, 10, 1):
            fields += ',audit_event_selector'
        params = {'name': self.parameters['name'], 'svm.name': self.parameters['vserver'], 'fields': fields}
        record, error = rest_generic.get_one_record(self.rest_api, api, params)
        if error:
            self.module.fail_json(msg='Error fetching S3 bucket %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        return self.form_current(record) if record else None

    def form_current(self, record):
        self.set_uuid(record)
        body = {'comment': self.na_helper.safe_get(record, ['comment']), 'size': self.na_helper.safe_get(record, ['size']), 'policy': self.na_helper.safe_get(record, ['policy']), 'qos_policy': self.na_helper.safe_get(record, ['qos_policy']), 'audit_event_selector': self.na_helper.safe_get(record, ['audit_event_selector']), 'type': self.na_helper.safe_get(record, ['type']), 'nas_path': self.na_helper.safe_get(record, ['nas_path'])}
        if body['policy']:
            for policy_statement in body['policy'].get('statements', []):
                policy_statement['sid'] = str(policy_statement['sid'])
                if not policy_statement.get('conditions'):
                    policy_statement['conditions'] = []
                else:
                    for condition in policy_statement['conditions']:
                        condition['delimiters'] = condition.get('delimiters')
                        condition['max_keys'] = condition.get('max_keys')
                        condition['operator'] = condition.get('operator')
                        condition['prefixes'] = condition.get('prefixes')
                        condition['source_ips'] = condition.get('source_ips')
                        condition['usernames'] = condition.get('usernames')
        else:
            body['policy'] = {'statements': []}
        return body

    def set_uuid(self, record):
        self.uuid = record['uuid']
        self.svm_uuid = record['svm']['uuid']
        self.volume_uuid = self.na_helper.safe_get(record, ['volume', 'uuid'])

    def create_s3_bucket(self):
        api = 'protocols/s3/buckets'
        body = {'svm.name': self.parameters['vserver'], 'name': self.parameters['name']}
        body.update(self.form_create_or_modify_body())
        dummy, error = rest_generic.post_async(self.rest_api, api, body, job_timeout=120)
        if error:
            self.module.fail_json(msg='Error creating S3 bucket %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def delete_s3_bucket(self):
        api = 'protocols/s3/buckets'
        uuids = '%s/%s' % (self.svm_uuid, self.uuid)
        dummy, error = rest_generic.delete_async(self.rest_api, api, uuids, job_timeout=120)
        if error:
            self.module.fail_json(msg='Error deleting S3 bucket %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def modify_s3_bucket(self, modify):
        api = 'protocols/s3/buckets'
        uuids = '%s/%s' % (self.svm_uuid, self.uuid)
        body = self.form_create_or_modify_body(modify)
        dummy, error = rest_generic.patch_async(self.rest_api, api, uuids, body, job_timeout=120)
        if error:
            self.module.fail_json(msg='Error modifying S3 bucket %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def form_create_or_modify_body(self, params=None):
        if params is None:
            params = self.parameters
        body = {}
        options = ['aggregates', 'constituents_per_aggregate', 'size', 'comment', 'type', 'nas_path', 'policy']
        for option in options:
            if option in params:
                body[option] = params[option]
        if 'qos_policy' in params:
            body['qos_policy'] = self.na_helper.filter_out_none_entries(params['qos_policy'])
        if 'audit_event_selector' in params:
            body['audit_event_selector'] = self.na_helper.filter_out_none_entries(params['audit_event_selector'])
        return body

    def check_volume_aggr(self):
        api = 'storage/volumes/%s' % self.volume_uuid
        params = {'fields': 'aggregates.name'}
        record, error = rest_generic.get_one_record(self.rest_api, api, params)
        if error:
            self.module.fail_json(msg=error)
        aggr_names = [aggr['name'] for aggr in record['aggregates']]
        if self.parameters.get('aggregates'):
            if sorted(aggr_names) != sorted(self.parameters['aggregates']):
                return True
        return False

    def validate_modify_required(self, modify, current):
        if len(modify['policy']['statements']) != len(current['policy']['statements']):
            return True
        match_found = []
        for statement in modify['policy']['statements']:
            for index, current_statement in enumerate(current['policy']['statements']):
                if index in match_found:
                    continue
                statement_modified = self.na_helper.get_modified_attributes(current_statement, statement)
                if not statement_modified:
                    match_found.append(index)
                    break
                if len(statement_modified) > 1:
                    continue
                if statement_modified.get('conditions'):
                    if not current_statement['conditions']:
                        continue
                    if len(statement_modified.get('conditions')) != len(current_statement['conditions']):
                        continue

                    def require_modify(desired, current):
                        for condition in desired:
                            if condition.get('operator'):
                                for current_condition in current:
                                    if condition['operator'] == current_condition['operator']:
                                        condition_modified = self.na_helper.get_modified_attributes(current_condition, condition)
                                        if condition_modified:
                                            return True
                            else:
                                return True
                    if not require_modify(statement_modified['conditions'], current_statement['conditions']):
                        match_found.append(index)
                        break
        return not match_found or len(match_found) != len(modify['policy']['statements'])

    def apply(self):
        current = self.get_s3_bucket()
        cd_action, modify = (None, None)
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if cd_action is None:
            modify = self.na_helper.get_modified_attributes(current, self.parameters)
            if modify.get('type'):
                self.module.fail_json(msg='Error: cannot modify bucket type.')
            if len(modify) == 1 and 'policy' in modify and current.get('policy'):
                if modify['policy'].get('statements'):
                    self.na_helper.changed = self.validate_modify_required(modify, current)
                    if not self.na_helper.changed:
                        modify = False
            if current and self.volume_uuid and self.check_volume_aggr():
                self.module.fail_json(msg='Aggregates cannot be modified for S3 bucket %s' % self.parameters['name'])
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_s3_bucket()
            if cd_action == 'delete':
                self.delete_s3_bucket()
            if modify:
                self.modify_s3_bucket(modify)
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify)
        self.module.exit_json(**result)