from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.composer.v1alpha2 import composer_v1alpha2_messages as messages
class ProjectsLocationsImageVersionsService(base_api.BaseApiService):
    """Service class for the projects_locations_imageVersions resource."""
    _NAME = 'projects_locations_imageVersions'

    def __init__(self, client):
        super(ComposerV1alpha2.ProjectsLocationsImageVersionsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """List ImageVersions for provided location.

      Args:
        request: (ComposerProjectsLocationsImageVersionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListImageVersionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/imageVersions', http_method='GET', method_id='composer.projects.locations.imageVersions.list', ordered_params=['parent'], path_params=['parent'], query_params=['includePastReleases', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/imageVersions', request_field='', request_type_name='ComposerProjectsLocationsImageVersionsListRequest', response_type_name='ListImageVersionsResponse', supports_download=False)