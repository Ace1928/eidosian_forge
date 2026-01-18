from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
class NetAppOntapObjectStoreConfig:
    """ object initialize and class methods """

    def __init__(self):
        self.use_rest = False
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, choices=['present', 'absent'], default='present'), change_password=dict(required=False, type='bool', default=False, no_log=False)))
        rest_options = dict(name=dict(required=True, type='str'), provider_type=dict(required=False, type='str'), server=dict(required=False, type='str'), container=dict(required=False, type='str'), access_key=dict(required=False, type='str', no_log=True), port=dict(required=False, type='int'), certificate_validation_enabled=dict(required=False, type='bool'), ssl_enabled=dict(required=False, type='bool'), owner=dict(required=False, type='str'))
        self.rest_get_fields = list(rest_options.keys())
        rest_options.update(dict(secret_password=dict(required=False, type='str', no_log=True)))
        self.rest_all_fields = rest_options.keys()
        self.argument_spec.update(rest_options)
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = OntapRestAPI(self.module)
        if self.rest_api.is_rest():
            self.use_rest = True
        elif not netapp_utils.has_netapp_lib():
            self.module.fail_json(msg=netapp_utils.netapp_lib_is_required())
        else:
            if self.parameters.get('owner', 'fabricpool') != 'fabricpool':
                self.module.fail_json(msg='Error: unsupported value for owner: %s when using ZAPI.' % self.parameters.get('owner'))
            self.server = netapp_utils.setup_na_ontap_zapi(module=self.module)

    def get_aggr_object_store(self):
        """
        Fetch details if object store config exists.
        :return:
            Dictionary of current details if object store config found
            None if object store config is not found
        """
        if self.use_rest:
            api = 'cloud/targets'
            query = {'name': self.parameters['name']}
            fields = ','.join(self.rest_get_fields)
            fields += ',uuid'
            record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
            if error:
                self.module.fail_json(msg='Error %s' % error)
            return record
        else:
            aggr_object_store_get_iter = netapp_utils.zapi.NaElement.create_node_with_children('aggr-object-store-config-get', **{'object-store-name': self.parameters['name']})
            try:
                result = self.server.invoke_successfully(aggr_object_store_get_iter, enable_tunneling=False)
            except netapp_utils.zapi.NaApiError as error:
                if to_native(error.code) == '15661':
                    return None
                else:
                    self.module.fail_json(msg=to_native(error), exception=traceback.format_exc())
            info = self.na_helper.safe_get(result, ['attributes', 'aggr-object-store-config-info'])
            if info:
                zapi_to_rest = {'access_key': dict(key_list=['access-key'], convert_to=str), 'certificate_validation_enabled': dict(key_list=['is-certificate-validation-enabled'], convert_to=bool), 'container': dict(key_list=['s3-name'], convert_to=str), 'name': dict(key_list=['object-store-name'], convert_to=str), 'port': dict(key_list=['port'], convert_to=int), 'provider_type': dict(key_list=['provider-type'], convert_to=str), 'ssl_enabled': dict(key_list=['ssl-enabled'], convert_to=bool), 'server': dict(key_list=['server'], convert_to=str)}
                results = {}
                self.na_helper.zapi_get_attrs(info, zapi_to_rest, results)
                return results
            return None

    def validate_and_build_body(self, modify=None):
        if modify is None:
            required_keys = set(['provider_type', 'server', 'container', 'access_key'])
            if not required_keys.issubset(set(self.parameters.keys())):
                self.module.fail_json(msg='Error provisioning object store %s: one of the following parameters are missing %s' % (self.parameters['name'], ', '.join(required_keys)))
        if not self.use_rest:
            return None
        params = self.parameters if modify is None else modify
        body = {}
        for key in self.rest_all_fields:
            if params.get(key) is not None:
                body[key] = params[key]
        if not modify and 'owner' not in body:
            body['owner'] = 'fabricpool'
        if modify and 'owner' in body:
            self.module.fail_json(msg='Error modifying object store, owner cannot be changed.  Found: %s.' % body['owner'])
        return body

    def create_aggr_object_store(self, body):
        """
        Create aggregate object store config
        :return: None
        """
        if self.use_rest:
            api = 'cloud/targets'
            dummy, error = rest_generic.post_async(self.rest_api, api, body)
            if error:
                self.module.fail_json(msg='Error %s' % error)
        else:
            options = {'object-store-name': self.parameters['name'], 'provider-type': self.parameters['provider_type'], 'server': self.parameters['server'], 's3-name': self.parameters['container'], 'access-key': self.parameters['access_key']}
            if self.parameters.get('secret_password'):
                options['secret-password'] = self.parameters['secret_password']
            if self.parameters.get('port') is not None:
                options['port'] = str(self.parameters['port'])
            if self.parameters.get('certificate_validation_enabled') is not None:
                options['is-certificate-validation-enabled'] = str(self.parameters['certificate_validation_enabled']).lower()
            if self.parameters.get('ssl_enabled') is not None:
                options['ssl-enabled'] = str(self.parameters['ssl_enabled']).lower()
            object_store_create = netapp_utils.zapi.NaElement.create_node_with_children('aggr-object-store-config-create', **options)
            try:
                self.server.invoke_successfully(object_store_create, enable_tunneling=False)
            except netapp_utils.zapi.NaApiError as error:
                self.module.fail_json(msg='Error provisioning object store config %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def modify_aggr_object_store(self, body, uuid=None):
        """
        modify aggregate object store config
        :return: None
        """
        api = 'cloud/targets'
        dummy, error = rest_generic.patch_async(self.rest_api, api, uuid, body)
        if error:
            self.module.fail_json(msg='Error %s' % error)

    def delete_aggr_object_store(self, uuid=None):
        """
        Delete aggregate object store config
        :return: None
        """
        if self.use_rest:
            api = 'cloud/targets'
            dummy, error = rest_generic.delete_async(self.rest_api, api, uuid)
            if error:
                self.module.fail_json(msg='Error %s' % error)
        else:
            object_store_destroy = netapp_utils.zapi.NaElement.create_node_with_children('aggr-object-store-config-delete', **{'object-store-name': self.parameters['name']})
            try:
                self.server.invoke_successfully(object_store_destroy, enable_tunneling=False)
            except netapp_utils.zapi.NaApiError as error:
                self.module.fail_json(msg='Error removing object store config %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def apply(self):
        """
        Apply action to the object store config
        :return: None
        """
        modify = None
        current = self.get_aggr_object_store()
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if cd_action is None:
            modify = self.na_helper.get_modified_attributes(current, self.parameters)
            if self.parameters['change_password'] and 'secret_password' in self.parameters:
                if not modify:
                    modify = {}
                modify['secret_password'] = self.parameters['secret_password']
                self.na_helper.changed = True
                self.module.warn('na_ontap_object_store is not idempotent when change_password is set to true')
        if not self.use_rest and modify:
            self.module.fail_json(msg='Error - modify is not supported with ZAPI: %s' % modify)
        if cd_action == 'create' or modify:
            body = self.validate_and_build_body(modify)
        if self.na_helper.changed and (not self.module.check_mode):
            uuid = current['uuid'] if current and self.use_rest else None
            if cd_action == 'create':
                self.create_aggr_object_store(body)
            elif cd_action == 'delete':
                self.delete_aggr_object_store(uuid)
            elif modify:
                self.modify_aggr_object_store(body, uuid)
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify)
        self.module.exit_json(**result)