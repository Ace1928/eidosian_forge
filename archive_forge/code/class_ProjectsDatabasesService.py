from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firestore.v1 import firestore_v1_messages as messages
class ProjectsDatabasesService(base_api.BaseApiService):
    """Service class for the projects_databases resource."""
    _NAME = 'projects_databases'

    def __init__(self, client):
        super(FirestoreV1.ProjectsDatabasesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a database.

      Args:
        request: (FirestoreProjectsDatabasesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases', http_method='POST', method_id='firestore.projects.databases.create', ordered_params=['parent'], path_params=['parent'], query_params=['databaseId'], relative_path='v1/{+parent}/databases', request_field='googleFirestoreAdminV1Database', request_type_name='FirestoreProjectsDatabasesCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a database.

      Args:
        request: (FirestoreProjectsDatabasesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}', http_method='DELETE', method_id='firestore.projects.databases.delete', ordered_params=['name'], path_params=['name'], query_params=['etag'], relative_path='v1/{+name}', request_field='', request_type_name='FirestoreProjectsDatabasesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def ExportDocuments(self, request, global_params=None):
        """Exports a copy of all or a subset of documents from Google Cloud Firestore to another storage system, such as Google Cloud Storage. Recent updates to documents may not be reflected in the export. The export occurs in the background and its progress can be monitored and managed via the Operation resource that is created. The output of an export may only be used once the associated operation is done. If an export operation is cancelled before completion it may leave partial data behind in Google Cloud Storage. For more details on export behavior and output format, refer to: https://cloud.google.com/firestore/docs/manage-data/export-import.

      Args:
        request: (FirestoreProjectsDatabasesExportDocumentsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('ExportDocuments')
        return self._RunMethod(config, request, global_params=global_params)
    ExportDocuments.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}:exportDocuments', http_method='POST', method_id='firestore.projects.databases.exportDocuments', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:exportDocuments', request_field='googleFirestoreAdminV1ExportDocumentsRequest', request_type_name='FirestoreProjectsDatabasesExportDocumentsRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information about a database.

      Args:
        request: (FirestoreProjectsDatabasesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleFirestoreAdminV1Database) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}', http_method='GET', method_id='firestore.projects.databases.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='FirestoreProjectsDatabasesGetRequest', response_type_name='GoogleFirestoreAdminV1Database', supports_download=False)

    def ImportDocuments(self, request, global_params=None):
        """Imports documents into Google Cloud Firestore. Existing documents with the same name are overwritten. The import occurs in the background and its progress can be monitored and managed via the Operation resource that is created. If an ImportDocuments operation is cancelled, it is possible that a subset of the data has already been imported to Cloud Firestore.

      Args:
        request: (FirestoreProjectsDatabasesImportDocumentsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('ImportDocuments')
        return self._RunMethod(config, request, global_params=global_params)
    ImportDocuments.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}:importDocuments', http_method='POST', method_id='firestore.projects.databases.importDocuments', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:importDocuments', request_field='googleFirestoreAdminV1ImportDocumentsRequest', request_type_name='FirestoreProjectsDatabasesImportDocumentsRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def List(self, request, global_params=None):
        """List all the databases in the project.

      Args:
        request: (FirestoreProjectsDatabasesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleFirestoreAdminV1ListDatabasesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases', http_method='GET', method_id='firestore.projects.databases.list', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/databases', request_field='', request_type_name='FirestoreProjectsDatabasesListRequest', response_type_name='GoogleFirestoreAdminV1ListDatabasesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a database.

      Args:
        request: (FirestoreProjectsDatabasesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}', http_method='PATCH', method_id='firestore.projects.databases.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleFirestoreAdminV1Database', request_type_name='FirestoreProjectsDatabasesPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Restore(self, request, global_params=None):
        """Creates a new database by restoring from an existing backup. The new database must be in the same cloud region or multi-region location as the existing backup. This behaves similar to FirestoreAdmin.CreateDatabase except instead of creating a new empty database, a new database is created with the database type, index configuration, and documents from an existing backup. The long-running operation can be used to track the progress of the restore, with the Operation's metadata field type being the RestoreDatabaseMetadata. The response type is the Database if the restore was successful. The new database is not readable or writeable until the LRO has completed.

      Args:
        request: (FirestoreProjectsDatabasesRestoreRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Restore')
        return self._RunMethod(config, request, global_params=global_params)
    Restore.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases:restore', http_method='POST', method_id='firestore.projects.databases.restore', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/databases:restore', request_field='googleFirestoreAdminV1RestoreDatabaseRequest', request_type_name='FirestoreProjectsDatabasesRestoreRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)