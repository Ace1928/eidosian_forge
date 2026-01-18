from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsSharedflowsRevisionsService(base_api.BaseApiService):
    """Service class for the organizations_environments_sharedflows_revisions resource."""
    _NAME = 'organizations_environments_sharedflows_revisions'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsSharedflowsRevisionsService, self).__init__(client)
        self._upload_configs = {}

    def Deploy(self, request, global_params=None):
        """Deploys a revision of a shared flow. If another revision of the same shared flow is currently deployed, set the `override` parameter to `true` to have this revision replace the currently deployed revision. You cannot use a shared flow until it has been deployed to an environment. For a request path `organizations/{org}/environments/{env}/sharedflows/{sf}/revisions/{rev}/deployments`, two permissions are required: * `apigee.deployments.create` on the resource `organizations/{org}/environments/{env}` * `apigee.sharedflowrevisions.deploy` on the resource `organizations/{org}/sharedflows/{sf}/revisions/{rev}`.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSharedflowsRevisionsDeployRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Deployment) The response message.
      """
        config = self.GetMethodConfig('Deploy')
        return self._RunMethod(config, request, global_params=global_params)
    Deploy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/sharedflows/{sharedflowsId}/revisions/{revisionsId}/deployments', http_method='POST', method_id='apigee.organizations.environments.sharedflows.revisions.deploy', ordered_params=['name'], path_params=['name'], query_params=['override', 'serviceAccount'], relative_path='v1/{+name}/deployments', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsSharedflowsRevisionsDeployRequest', response_type_name='GoogleCloudApigeeV1Deployment', supports_download=False)

    def GetDeployments(self, request, global_params=None):
        """Gets the deployment of a shared flow revision and actual state reported by runtime pods.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSharedflowsRevisionsGetDeploymentsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Deployment) The response message.
      """
        config = self.GetMethodConfig('GetDeployments')
        return self._RunMethod(config, request, global_params=global_params)
    GetDeployments.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/sharedflows/{sharedflowsId}/revisions/{revisionsId}/deployments', http_method='GET', method_id='apigee.organizations.environments.sharedflows.revisions.getDeployments', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}/deployments', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsSharedflowsRevisionsGetDeploymentsRequest', response_type_name='GoogleCloudApigeeV1Deployment', supports_download=False)

    def Undeploy(self, request, global_params=None):
        """Undeploys a shared flow revision from an environment. For a request path `organizations/{org}/environments/{env}/sharedflows/{sf}/revisions/{rev}/deployments`, two permissions are required: * `apigee.deployments.delete` on the resource `organizations/{org}/environments/{env}` * `apigee.sharedflowrevisions.undeploy` on the resource `organizations/{org}/sharedflows/{sf}/revisions/{rev}`.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSharedflowsRevisionsUndeployRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Undeploy')
        return self._RunMethod(config, request, global_params=global_params)
    Undeploy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/sharedflows/{sharedflowsId}/revisions/{revisionsId}/deployments', http_method='DELETE', method_id='apigee.organizations.environments.sharedflows.revisions.undeploy', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}/deployments', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsSharedflowsRevisionsUndeployRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)