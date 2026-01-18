from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v1 import iam_v1_messages as messages
class LocationsWorkforcePoolsSubjectsService(base_api.BaseApiService):
    """Service class for the locations_workforcePools_subjects resource."""
    _NAME = 'locations_workforcePools_subjects'

    def __init__(self, client):
        super(IamV1.LocationsWorkforcePoolsSubjectsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a WorkforcePoolSubject. Subject must not already be in a deleted state. A WorkforcePoolSubject is automatically created the first time an external credential is exchanged for a Google Cloud credential with a mapped `google.subject` attribute. There is no path to manually create WorkforcePoolSubjects. Once deleted, the WorkforcePoolSubject may not be used for 30 days. After 30 days, the WorkforcePoolSubject will be deleted forever and can be reused in token exchanges with Google Cloud STS. This will automatically create a new WorkforcePoolSubject that is independent of the previously deleted WorkforcePoolSubject with the same google.subject value.

      Args:
        request: (IamLocationsWorkforcePoolsSubjectsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}/subjects/{subjectsId}', http_method='DELETE', method_id='iam.locations.workforcePools.subjects.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IamLocationsWorkforcePoolsSubjectsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Undelete(self, request, global_params=None):
        """Undeletes a WorkforcePoolSubject, as long as it was deleted fewer than 30 days ago.

      Args:
        request: (IamLocationsWorkforcePoolsSubjectsUndeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Undelete')
        return self._RunMethod(config, request, global_params=global_params)
    Undelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/locations/{locationsId}/workforcePools/{workforcePoolsId}/subjects/{subjectsId}:undelete', http_method='POST', method_id='iam.locations.workforcePools.subjects.undelete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:undelete', request_field='undeleteWorkforcePoolSubjectRequest', request_type_name='IamLocationsWorkforcePoolsSubjectsUndeleteRequest', response_type_name='Operation', supports_download=False)