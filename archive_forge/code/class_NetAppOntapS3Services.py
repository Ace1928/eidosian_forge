from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppOntapS3Services:

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), name=dict(required=True, type='str'), enabled=dict(required=False, type='bool'), vserver=dict(required=True, type='str'), comment=dict(required=False, type='str'), certificate_name=dict(required=False, type='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.svm_uuid = None
        self.na_helper = NetAppModule(self.module)
        self.parameters = self.na_helper.check_and_set_parameters(self.module)
        self.rest_api = OntapRestAPI(self.module)
        self.use_rest = self.rest_api.is_rest()
        self.rest_api.fail_if_not_rest_minimum_version('na_ontap_s3_services', 9, 8)

    def get_s3_service(self, extra_field=False):
        api = 'protocols/s3/services'
        fields = ','.join(('name', 'enabled', 'svm.uuid', 'comment', 'certificate.name'))
        if extra_field:
            fields += ',users'
        params = {'name': self.parameters['name'], 'svm.name': self.parameters['vserver'], 'fields': fields}
        record, error = rest_generic.get_one_record(self.rest_api, api, params)
        if error:
            self.module.fail_json(msg='Error fetching S3 service %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
        if record:
            if self.na_helper.safe_get(record, ['certificate', 'name']):
                record['certificate_name'] = self.na_helper.safe_get(record, ['certificate', 'name'])
            return self.set_uuids(record)
        return None

    def create_s3_service(self):
        api = 'protocols/s3/services'
        body = {'svm.name': self.parameters['vserver'], 'name': self.parameters['name']}
        if self.parameters.get('enabled') is not None:
            body['enabled'] = self.parameters['enabled']
        if self.parameters.get('comment'):
            body['comment'] = self.parameters['comment']
        if self.parameters.get('certificate_name'):
            body['certificate.name'] = self.parameters['certificate_name']
        dummy, error = rest_generic.post_async(self.rest_api, api, body)
        if error:
            self.module.fail_json(msg='Error creating S3 service %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def delete_s3_service(self):
        api = 'protocols/s3/services'
        body = {'delete_all': False}
        dummy, error = rest_generic.delete_async(self.rest_api, api, self.svm_uuid, body=body)
        if error:
            self.module.fail_json(msg='Error deleting S3 service %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def modify_s3_service(self, modify):
        api = 'protocols/s3/services'
        body = {'name': self.parameters['name']}
        if modify.get('enabled') is not None:
            body['enabled'] = self.parameters['enabled']
        if modify.get('comment'):
            body['comment'] = self.parameters['comment']
        if modify.get('certificate_name'):
            body['certificate.name'] = self.parameters['certificate_name']
        dummy, error = rest_generic.patch_async(self.rest_api, api, self.svm_uuid, body)
        if error:
            self.module.fail_json(msg='Error modifying S3 service %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def set_uuids(self, record):
        self.svm_uuid = record['svm']['uuid']
        return record

    def parse_response(self, response):
        if response is not None:
            users_info = []
            options = ['name', 'access_key', 'secret_key']
            for user_info in response.get('users'):
                info = {}
                for option in options:
                    if user_info.get(option) is not None:
                        info[option] = user_info.get(option)
                users_info.append(info)
            return {'name': response.get('name'), 'enabled': response.get('enabled'), 'certificate_name': response.get('certificate_name'), 'users': users_info, 'svm': {'name': self.na_helper.safe_get(response, ['svm', 'name']), 'uuid': self.na_helper.safe_get(response, ['svm', 'uuid'])}}
        return None

    def apply(self):
        current = self.get_s3_service()
        cd_action, modify, response = (None, None, None)
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if cd_action is None:
            modify = self.na_helper.get_modified_attributes(current, self.parameters)
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_s3_service()
                response = self.get_s3_service(True)
            if cd_action == 'delete':
                self.delete_s3_service()
            if modify:
                self.modify_s3_service(modify)
                response = self.get_s3_service(True)
        message = self.parse_response(response)
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify, extra_responses={'s3_service_info': message})
        self.module.exit_json(**result)