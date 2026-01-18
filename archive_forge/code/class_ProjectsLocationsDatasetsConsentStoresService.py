from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1alpha2 import healthcare_v1alpha2_messages as messages
class ProjectsLocationsDatasetsConsentStoresService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_consentStores resource."""
    _NAME = 'projects_locations_datasets_consentStores'

    def __init__(self, client):
        super(HealthcareV1alpha2.ProjectsLocationsDatasetsConsentStoresService, self).__init__(client)
        self._upload_configs = {}

    def CheckDataAccess(self, request, global_params=None):
        """Checks if a particular data_id of a User data mapping in the specified consent store is consented for the specified use.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresCheckDataAccessRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CheckDataAccessResponse) The response message.
      """
        config = self.GetMethodConfig('CheckDataAccess')
        return self._RunMethod(config, request, global_params=global_params)
    CheckDataAccess.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}:checkDataAccess', http_method='POST', method_id='healthcare.projects.locations.datasets.consentStores.checkDataAccess', ordered_params=['consentStore'], path_params=['consentStore'], query_params=[], relative_path='v1alpha2/{+consentStore}:checkDataAccess', request_field='checkDataAccessRequest', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresCheckDataAccessRequest', response_type_name='CheckDataAccessResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new consent store in the parent dataset. Attempting to create a consent store with the same ID as an existing store fails with an ALREADY_EXISTS error.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConsentStore) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores', http_method='POST', method_id='healthcare.projects.locations.datasets.consentStores.create', ordered_params=['parent'], path_params=['parent'], query_params=['consentStoreId'], relative_path='v1alpha2/{+parent}/consentStores', request_field='consentStore', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresCreateRequest', response_type_name='ConsentStore', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified consent store and removes all the consent store's data.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}', http_method='DELETE', method_id='healthcare.projects.locations.datasets.consentStores.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresDeleteRequest', response_type_name='Empty', supports_download=False)

    def EvaluateUserConsents(self, request, global_params=None):
        """Evaluates the end user's Consents for all matching User data mappings. Note: User data mappings are indexed asynchronously, which can cause a slight delay between the time mappings are created or updated and when they are included in EvaluateUserConsents results.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresEvaluateUserConsentsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EvaluateUserConsentsResponse) The response message.
      """
        config = self.GetMethodConfig('EvaluateUserConsents')
        return self._RunMethod(config, request, global_params=global_params)
    EvaluateUserConsents.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}:evaluateUserConsents', http_method='POST', method_id='healthcare.projects.locations.datasets.consentStores.evaluateUserConsents', ordered_params=['consentStore'], path_params=['consentStore'], query_params=[], relative_path='v1alpha2/{+consentStore}:evaluateUserConsents', request_field='evaluateUserConsentsRequest', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresEvaluateUserConsentsRequest', response_type_name='EvaluateUserConsentsResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the specified consent store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConsentStore) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}', http_method='GET', method_id='healthcare.projects.locations.datasets.consentStores.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresGetRequest', response_type_name='ConsentStore', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}:getIamPolicy', http_method='GET', method_id='healthcare.projects.locations.datasets.consentStores.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha2/{+resource}:getIamPolicy', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the consent stores in the given dataset.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListConsentStoresResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores', http_method='GET', method_id='healthcare.projects.locations.datasets.consentStores.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/consentStores', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresListRequest', response_type_name='ListConsentStoresResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified consent store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConsentStore) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}', http_method='PATCH', method_id='healthcare.projects.locations.datasets.consentStores.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha2/{+name}', request_field='consentStore', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresPatchRequest', response_type_name='ConsentStore', supports_download=False)

    def QueryAccessibleData(self, request, global_params=None):
        """Queries all data_ids that are consented for a specified use in the given consent store and writes them to a specified destination. The returned Operation includes a progress counter for the number of User data mappings processed. If the request is successful, a detailed response is returned of type QueryAccessibleDataResponse, contained in the response field when the operation finishes. The metadata field type is OperationMetadata. Errors are logged to Cloud Logging (see [Viewing error logs in Cloud Logging](https://cloud.google.com/healthcare/docs/how-tos/logging)). For example, the following sample log entry shows a `failed to evaluate consent policy` error that occurred during a QueryAccessibleData call to consent store `projects/{project_id}/locations/{location_id}/datasets/{dataset_id}/consentStores/{consent_store_id}`. ```json jsonPayload: { @type: "type.googleapis.com/google.cloud.healthcare.logging.QueryAccessibleDataLogEntry" error: { code: 9 message: "failed to evaluate consent policy" } resourceName: "projects/{project_id}/locations/{location_id}/datasets/{dataset_id}/consentStores/{consent_store_id}/consents/{consent_id}" } logName: "projects/{project_id}/logs/healthcare.googleapis.com%2Fquery_accessible_data" operation: { id: "projects/{project_id}/locations/{location_id}/datasets/{dataset_id}/operations/{operation_id}" producer: "healthcare.googleapis.com/QueryAccessibleData" } receiveTimestamp: "TIMESTAMP" resource: { labels: { consent_store_id: "{consent_store_id}" dataset_id: "{dataset_id}" location: "{location_id}" project_id: "{project_id}" } type: "healthcare_consent_store" } severity: "ERROR" timestamp: "TIMESTAMP" ```.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresQueryAccessibleDataRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('QueryAccessibleData')
        return self._RunMethod(config, request, global_params=global_params)
    QueryAccessibleData.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}:queryAccessibleData', http_method='POST', method_id='healthcare.projects.locations.datasets.consentStores.queryAccessibleData', ordered_params=['consentStore'], path_params=['consentStore'], query_params=[], relative_path='v1alpha2/{+consentStore}:queryAccessibleData', request_field='queryAccessibleDataRequest', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresQueryAccessibleDataRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}:setIamPolicy', http_method='POST', method_id='healthcare.projects.locations.datasets.consentStores.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}:testIamPermissions', http_method='POST', method_id='healthcare.projects.locations.datasets.consentStores.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)