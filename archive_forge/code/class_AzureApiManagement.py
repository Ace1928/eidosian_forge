from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_rest import GenericRestClient
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
import json
class AzureApiManagement(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', updatable=False, disposition='resourceGroupName', required=True), service_name=dict(type='str', updatable=False, disposition='serviceName', required=True), api_id=dict(type='str', updatable=False, disposition='apiId', required=True), description=dict(type='str', disposition='/properties/description'), authentication_settings=dict(type='dict', disposition='/properties/authenticationSettings', options=dict(o_auth2=dict(type='dict', disposition='oAuth2', options=dict(authorization_server_id=dict(type='str', disposition='authorizationServerId'), scope=dict(type='str', disposition='scope'))), openid=dict(type='dict', options=dict(openid_provider_id=dict(type='str', disposition='openidProviderId'), bearer_token_sending_methods=dict(type='list', elements='str', disposition='bearerTokenSendingMethods', choices=['authorizationHeader', 'query']))))), subscription_key_parameter_names=dict(type='dict', no_log=True, disposition='/properties/subscriptionKeyParameterNames', options=dict(header=dict(type='str', required=False), query=dict(type='str', required=False))), type=dict(type='str', disposition='/properties/type', choices=['http', 'soap']), api_revision=dict(type='str', disposition='/properties/apiRevision'), api_version=dict(type='str', disposition='/properties/apiVersion'), is_current=dict(type='bool', disposition='/properties/isCurrent'), api_revision_description=dict(type='str', disposition='/properties/apiRevisionDescription'), api_version_description=dict(type='str', disposition='/properties/apiVersionDescription'), api_version_set_id=dict(type='str', disposition='/properties/apiVersionSetId'), subscription_required=dict(type='bool', disposition='/properties/subscriptionRequired'), source_api_id=dict(type='str', disposition='/properties/sourceApiId'), display_name=dict(type='str', disposition='/properties/displayName'), service_url=dict(type='str', disposition='/properties/serviceUrl'), path=dict(type='str', disposition='/properties/*'), protocols=dict(type='list', elements='str', disposition='/properties/protocols', choices=['http', 'https']), api_version_set=dict(type='dict', disposition='/properties/apiVersionSet', options=dict(id=dict(type='str'), name=dict(type='str'), description=dict(type='str'), versioning_scheme=dict(type='str', disposition='versioningScheme', choices=['Segment', 'Query', 'Header']), version_query_name=dict(type='str', disposition='versionQueryName'), version_header_name=dict(type='str', disposition='versionHeaderName'))), value=dict(type='str', disposition='/properties/*'), format=dict(type='str', disposition='/properties/*', choices=['wadl-xml', 'wadl-link-json', 'swagger-json', 'swagger-link-json', 'wsdl', 'wsdl-link', 'openapi', 'openapi+json', 'openapi-link']), wsdl_selector=dict(type='dict', disposition='/properties/wsdlSelector', options=dict(wsdl_service_name=dict(type='str', disposition='wsdlServiceName'), wsdl_endpoint_name=dict(type='str', disposition='wsdlEndpointName'))), api_type=dict(type='str', disposition='/properties/apiType', choices=['http', 'soap']), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.service_name = None
        self.api_id = None
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.state = None
        self.url = None
        self.status_code = [200, 201, 202]
        self.to_do = Actions.NoAction
        self.body = {}
        self.query_parameters = {}
        self.query_parameters['api-version'] = '2020-06-01-preview'
        self.header_parameters = {}
        self.header_parameters['Content-Type'] = 'application/json; charset=utf-8'
        super(AzureApiManagement, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def get_url(self):
        return '/subscriptions' + '/' + self.subscription_id + '/resourceGroups' + '/' + self.resource_group + '/providers' + '/Microsoft.ApiManagement' + '/service' + '/' + self.service_name + '/apis' + '/' + self.api_id

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.body[key] = kwargs[key]
        self.inflate_parameters(self.module_arg_spec, self.body, 0)
        self.url = self.get_url()
        old_response = None
        response = None
        self.mgmt_client = self.get_mgmt_svc_client(GenericRestClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        old_response = self.get_resource()
        if not old_response:
            self.log('Api instance does not exist in the given service.')
            if self.state == 'present':
                self.to_do = Actions.Create
            else:
                self.log("Old instance didn't exist")
        else:
            self.log('Api instance already exists in the given service.')
            if self.state == 'absent':
                self.to_do = Actions.Delete
            else:
                modifiers = {}
                self.create_compare_modifiers(self.module_arg_spec, '', modifiers)
                self.results['modifiers'] = modifiers
                self.results['compare'] = []
                if not self.default_compare(modifiers, self.body, old_response, '', self.results):
                    self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.log('Create and Update the Api instance.')
            if self.check_mode:
                self.results['changed'] = True
                return self.results
            response = self.create_and_update_resource()
            self.results['changed'] = True
        elif self.to_do == Actions.Delete:
            self.log('Api instance deleted.')
            if self.check_mode:
                self.results['changed'] = True
                return self.results
            self.delete_resource()
            self.results['changed'] = True
        else:
            self.log('No change in Api instance.')
            self.results['changed'] = False
            response = old_response
        if response:
            self.results['id'] = response['id']
        return self.results

    def create_and_update_resource(self):
        try:
            response = self.mgmt_client.query(self.url, 'PUT', self.query_parameters, self.header_parameters, self.body, self.status_code, 600, 30)
        except Exception as exc:
            self.log('Error while creating/updating the Api instance.')
            self.fail('Error creating the Api instance: {0}'.format(str(exc)))
        try:
            response = json.loads(response.body())
        except Exception:
            response = {'text': response.context['deserialized_data']}
        return response

    def delete_resource(self):
        isDeleted = False
        try:
            response = self.mgmt_client.query(self.url, 'DELETE', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
            isDeleted = True
        except Exception as e:
            self.log('Error attempting to delete the Api instance.')
            self.fail('Error deleting the Api instance: {0}'.format(str(e)))
        return isDeleted

    def get_resource(self):
        isFound = False
        try:
            response = self.mgmt_client.query(self.url, 'GET', self.query_parameters, self.header_parameters, None, self.status_code, 600, 30)
            isFound = True
            response = json.loads(response.body())
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.log('Could not find the Api instance from the given parameters.')
        if isFound is True:
            return response
        return False