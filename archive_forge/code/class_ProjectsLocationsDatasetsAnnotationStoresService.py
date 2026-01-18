from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1alpha2 import healthcare_v1alpha2_messages as messages
class ProjectsLocationsDatasetsAnnotationStoresService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_annotationStores resource."""
    _NAME = 'projects_locations_datasets_annotationStores'

    def __init__(self, client):
        super(HealthcareV1alpha2.ProjectsLocationsDatasetsAnnotationStoresService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Annotation store within the parent dataset.

      Args:
        request: (HealthcareProjectsLocationsDatasetsAnnotationStoresCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnnotationStore) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/annotationStores', http_method='POST', method_id='healthcare.projects.locations.datasets.annotationStores.create', ordered_params=['parent'], path_params=['parent'], query_params=['annotationStoreId'], relative_path='v1alpha2/{+parent}/annotationStores', request_field='annotationStore', request_type_name='HealthcareProjectsLocationsDatasetsAnnotationStoresCreateRequest', response_type_name='AnnotationStore', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified Annotation store and removes all annotations that are contained within it.

      Args:
        request: (HealthcareProjectsLocationsDatasetsAnnotationStoresDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/annotationStores/{annotationStoresId}', http_method='DELETE', method_id='healthcare.projects.locations.datasets.annotationStores.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsAnnotationStoresDeleteRequest', response_type_name='Empty', supports_download=False)

    def Evaluate(self, request, global_params=None):
        """Evaluate an Annotation store against a ground truth Annotation store. When the operation finishes successfully, a detailed response is returned of type EvaluateAnnotationStoreResponse, contained in the response. The metadata field type is OperationMetadata. Errors are logged to Cloud Logging (see [Viewing error logs in Cloud Logging](https://cloud.google.com/healthcare/docs/how-tos/logging) and ImportAnnotations for a sample log entry).

      Args:
        request: (HealthcareProjectsLocationsDatasetsAnnotationStoresEvaluateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Evaluate')
        return self._RunMethod(config, request, global_params=global_params)
    Evaluate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/annotationStores/{annotationStoresId}:evaluate', http_method='POST', method_id='healthcare.projects.locations.datasets.annotationStores.evaluate', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:evaluate', request_field='evaluateAnnotationStoreRequest', request_type_name='HealthcareProjectsLocationsDatasetsAnnotationStoresEvaluateRequest', response_type_name='Operation', supports_download=False)

    def Export(self, request, global_params=None):
        """Export Annotations from the Annotation store. If the request is successful, a detailed response is returned of type ExportAnnotationsResponse, contained in the response field when the operation finishes. The metadata field type is OperationMetadata. Errors are logged to Cloud Logging (see [Viewing error logs in Cloud Logging](https://cloud.google.com/healthcare/docs/how-tos/logging) and ImportAnnotations for a sample log entry).

      Args:
        request: (HealthcareProjectsLocationsDatasetsAnnotationStoresExportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Export')
        return self._RunMethod(config, request, global_params=global_params)
    Export.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/annotationStores/{annotationStoresId}:export', http_method='POST', method_id='healthcare.projects.locations.datasets.annotationStores.export', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:export', request_field='exportAnnotationsRequest', request_type_name='HealthcareProjectsLocationsDatasetsAnnotationStoresExportRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the specified Annotation store or returns NOT_FOUND if it does not exist.

      Args:
        request: (HealthcareProjectsLocationsDatasetsAnnotationStoresGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnnotationStore) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/annotationStores/{annotationStoresId}', http_method='GET', method_id='healthcare.projects.locations.datasets.annotationStores.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsAnnotationStoresGetRequest', response_type_name='AnnotationStore', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (HealthcareProjectsLocationsDatasetsAnnotationStoresGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/annotationStores/{annotationStoresId}:getIamPolicy', http_method='GET', method_id='healthcare.projects.locations.datasets.annotationStores.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha2/{+resource}:getIamPolicy', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsAnnotationStoresGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Import(self, request, global_params=None):
        """Import Annotations to the Annotation store by loading data from the specified sources. If the request is successful, a detailed response is returned as of type ImportAnnotationsResponse, contained in the response field when the operation finishes. The metadata field type is OperationMetadata. Errors are logged to Cloud Logging (see [Viewing error logs in Cloud Logging](https://cloud.google.com/healthcare/docs/how-tos/logging)). For example, the following sample log entry shows a `failed to parse Cloud Storage object` error that occurred while attempting to import `gs://ANNOTATION_FILENAME.json` to `projects/{project_id}/locations/{location_id}/datasets/{dataset_id}/annotationStores/{annotation_store_id}`. ```json jsonPayload: { @type: "type.googleapis.com/google.cloud.healthcare.logging.ImportAnnotationLogEntry" error: { code: 3 message: "failed to parse Cloud Storage object" } source: "gs://ANNOTATION_FILENAME.json" } logName: "projects/{project_id}/logs/healthcare.googleapis.com%2Fimport_annotations" operation: { id: "projects/{project_id}/locations/{location_id}/datasets/{dataset_id}/operations/{operation_id}" producer: "healthcare.googleapis.com/ImportAnnotations" } receiveTimestamp: "TIMESTAMP" resource: { labels: { annotation_store_id: "{annotation_store_id}" dataset_id: "{dataset_id}" location: "{location_id}" project_id: "{project_id}" } type: "healthcare_annotation_store" } severity: "ERROR" timestamp: "TIMESTAMP" ```.

      Args:
        request: (HealthcareProjectsLocationsDatasetsAnnotationStoresImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Import')
        return self._RunMethod(config, request, global_params=global_params)
    Import.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/annotationStores/{annotationStoresId}:import', http_method='POST', method_id='healthcare.projects.locations.datasets.annotationStores.import', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:import', request_field='importAnnotationsRequest', request_type_name='HealthcareProjectsLocationsDatasetsAnnotationStoresImportRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the Annotation stores in the given dataset for a source store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsAnnotationStoresListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAnnotationStoresResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/annotationStores', http_method='GET', method_id='healthcare.projects.locations.datasets.annotationStores.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/annotationStores', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsAnnotationStoresListRequest', response_type_name='ListAnnotationStoresResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified Annotation store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsAnnotationStoresPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnnotationStore) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/annotationStores/{annotationStoresId}', http_method='PATCH', method_id='healthcare.projects.locations.datasets.annotationStores.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha2/{+name}', request_field='annotationStore', request_type_name='HealthcareProjectsLocationsDatasetsAnnotationStoresPatchRequest', response_type_name='AnnotationStore', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (HealthcareProjectsLocationsDatasetsAnnotationStoresSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/annotationStores/{annotationStoresId}:setIamPolicy', http_method='POST', method_id='healthcare.projects.locations.datasets.annotationStores.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='HealthcareProjectsLocationsDatasetsAnnotationStoresSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (HealthcareProjectsLocationsDatasetsAnnotationStoresTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/annotationStores/{annotationStoresId}:testIamPermissions', http_method='POST', method_id='healthcare.projects.locations.datasets.annotationStores.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='HealthcareProjectsLocationsDatasetsAnnotationStoresTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)