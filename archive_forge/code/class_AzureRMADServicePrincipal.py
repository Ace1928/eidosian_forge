from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
class AzureRMADServicePrincipal(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(app_id=dict(type='str', required=True), tenant=dict(type='str', required=True), state=dict(type='str', default='present', choices=['present', 'absent']), app_role_assignment_required=dict(type='bool'))
        self.state = None
        self.tenant = None
        self.app_id = None
        self.app_role_assignment_required = None
        self.object_id = None
        self.results = dict(changed=False)
        super(AzureRMADServicePrincipal, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=False, supports_tags=False, is_ad_resource=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            setattr(self, key, kwargs[key])
        response = self.get_resource()
        if response:
            if self.state == 'present':
                if self.check_update(response):
                    self.update_resource(response)
            elif self.state == 'absent':
                self.delete_resource(response)
        elif self.state == 'present':
            self.create_resource()
        elif self.state == 'absent':
            self.log('try to delete non exist resource')
        return self.results

    def create_resource(self):
        try:
            client = self.get_graphrbac_client(self.tenant)
            response = client.service_principals.create(ServicePrincipalCreateParameters(app_id=self.app_id, account_enabled=True))
            self.results['changed'] = True
            self.results.update(self.to_dict(response))
            return response
        except GraphErrorException as ge:
            self.fail('Error creating service principle, app id {0} - {1}'.format(self.app_id, str(ge)))

    def update_resource(self, old_response):
        try:
            client = self.get_graphrbac_client(self.tenant)
            to_update = {}
            if self.app_role_assignment_required is not None:
                to_update['app_role_assignment_required'] = self.app_role_assignment_required
            client.service_principals.update(old_response['object_id'], to_update)
            self.results['changed'] = True
            self.results.update(self.get_resource())
        except GraphErrorException as ge:
            self.fail('Error updating the service principal app_id {0} - {1}'.format(self.app_id, str(ge)))

    def delete_resource(self, response):
        try:
            client = self.get_graphrbac_client(self.tenant)
            client.service_principals.delete(response.get('object_id'))
            self.results['changed'] = True
            return True
        except GraphErrorException as ge:
            self.fail('Error deleting service principal app_id {0} - {1}'.format(self.app_id, str(ge)))

    def get_resource(self):
        try:
            client = self.get_graphrbac_client(self.tenant)
            result = list(client.service_principals.list(filter="servicePrincipalNames/any(c:c eq '{0}')".format(self.app_id)))
            if not result:
                return False
            result = result[0]
            return self.to_dict(result)
        except GraphErrorException as ge:
            self.log('Did not find the graph instance instance {0} - {1}'.format(self.app_id, str(ge)))
            return False

    def check_update(self, response):
        app_assignment_changed = self.app_role_assignment_required is not None and self.app_role_assignment_required != response.get('app_role_assignment_required', None)
        return app_assignment_changed

    def to_dict(self, object):
        return dict(app_id=object.app_id, object_id=object.object_id, app_display_name=object.display_name, app_role_assignment_required=object.app_role_assignment_required)