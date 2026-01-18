from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkemulticloud.v1 import gkemulticloud_v1_messages as messages
class ProjectsLocationsAwsClustersWellKnownService(base_api.BaseApiService):
    """Service class for the projects_locations_awsClusters_well_known resource."""
    _NAME = 'projects_locations_awsClusters_well_known'

    def __init__(self, client):
        super(GkemulticloudV1.ProjectsLocationsAwsClustersWellKnownService, self).__init__(client)
        self._upload_configs = {}

    def GetOpenid_configuration(self, request, global_params=None):
        """Gets the OIDC discovery document for the cluster. See the [OpenID Connect Discovery 1.0 specification](https://openid.net/specs/openid-connect-discovery-1_0.html) for details.

      Args:
        request: (GkemulticloudProjectsLocationsAwsClustersWellKnownGetOpenidConfigurationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGkemulticloudV1AwsOpenIdConfig) The response message.
      """
        config = self.GetMethodConfig('GetOpenid_configuration')
        return self._RunMethod(config, request, global_params=global_params)
    GetOpenid_configuration.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/awsClusters/{awsClustersId}/.well-known/openid-configuration', http_method='GET', method_id='gkemulticloud.projects.locations.awsClusters.well-known.getOpenid-configuration', ordered_params=['awsCluster'], path_params=['awsCluster'], query_params=[], relative_path='v1/{+awsCluster}/.well-known/openid-configuration', request_field='', request_type_name='GkemulticloudProjectsLocationsAwsClustersWellKnownGetOpenidConfigurationRequest', response_type_name='GoogleCloudGkemulticloudV1AwsOpenIdConfig', supports_download=False)