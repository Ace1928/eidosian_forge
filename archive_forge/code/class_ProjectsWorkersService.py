from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.genomics.v2alpha1 import genomics_v2alpha1_messages as messages
class ProjectsWorkersService(base_api.BaseApiService):
    """Service class for the projects_workers resource."""
    _NAME = 'projects_workers'

    def __init__(self, client):
        super(GenomicsV2alpha1.ProjectsWorkersService, self).__init__(client)
        self._upload_configs = {}

    def CheckIn(self, request, global_params=None):
        """The worker uses this method to retrieve the assigned operation and provide periodic status updates.

      Args:
        request: (GenomicsProjectsWorkersCheckInRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CheckInResponse) The response message.
      """
        config = self.GetMethodConfig('CheckIn')
        return self._RunMethod(config, request, global_params=global_params)
    CheckIn.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2alpha1/projects/{projectsId}/workers/{workersId}:checkIn', http_method='POST', method_id='genomics.projects.workers.checkIn', ordered_params=['id'], path_params=['id'], query_params=[], relative_path='v2alpha1/{+id}:checkIn', request_field='checkInRequest', request_type_name='GenomicsProjectsWorkersCheckInRequest', response_type_name='CheckInResponse', supports_download=False)