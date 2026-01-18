from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datacatalog.v1 import datacatalog_v1_messages as messages
class ProjectsLocationsEntryGroupsEntriesService(base_api.BaseApiService):
    """Service class for the projects_locations_entryGroups_entries resource."""
    _NAME = 'projects_locations_entryGroups_entries'

    def __init__(self, client):
        super(DatacatalogV1.ProjectsLocationsEntryGroupsEntriesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an entry. You can create entries only with 'FILESET', 'CLUSTER', 'DATA_STREAM', or custom types. Data Catalog automatically creates entries with other types during metadata ingestion from integrated systems. You must enable the Data Catalog API in the project identified by the `parent` parameter. For more information, see [Data Catalog resource project](https://cloud.google.com/data-catalog/docs/concepts/resource-project). An entry group can have a maximum of 100,000 entries.

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsEntriesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1Entry) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups/{entryGroupsId}/entries', http_method='POST', method_id='datacatalog.projects.locations.entryGroups.entries.create', ordered_params=['parent'], path_params=['parent'], query_params=['entryId'], relative_path='v1/{+parent}/entries', request_field='googleCloudDatacatalogV1Entry', request_type_name='DatacatalogProjectsLocationsEntryGroupsEntriesCreateRequest', response_type_name='GoogleCloudDatacatalogV1Entry', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an existing entry. You can delete only the entries created by the CreateEntry method. You must enable the Data Catalog API in the project identified by the `name` parameter. For more information, see [Data Catalog resource project](https://cloud.google.com/data-catalog/docs/concepts/resource-project).

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsEntriesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups/{entryGroupsId}/entries/{entriesId}', http_method='DELETE', method_id='datacatalog.projects.locations.entryGroups.entries.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DatacatalogProjectsLocationsEntryGroupsEntriesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an entry.

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsEntriesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1Entry) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups/{entryGroupsId}/entries/{entriesId}', http_method='GET', method_id='datacatalog.projects.locations.entryGroups.entries.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DatacatalogProjectsLocationsEntryGroupsEntriesGetRequest', response_type_name='GoogleCloudDatacatalogV1Entry', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May return: * A`NOT_FOUND` error if the resource doesn't exist or you don't have the permission to view it. * An empty policy if the resource exists but doesn't have a set policy. Supported resources are: - Tag templates - Entry groups Note: This method doesn't get policies from Google Cloud Platform resources ingested into Data Catalog. To call this method, you must have the following Google IAM permissions: - `datacatalog.tagTemplates.getIamPolicy` to get policies on tag templates. - `datacatalog.entryGroups.getIamPolicy` to get policies on entry groups.

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsEntriesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups/{entryGroupsId}/entries/{entriesId}:getIamPolicy', http_method='POST', method_id='datacatalog.projects.locations.entryGroups.entries.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='DatacatalogProjectsLocationsEntryGroupsEntriesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Import(self, request, global_params=None):
        """Imports entries from a source, such as data previously dumped into a Cloud Storage bucket, into Data Catalog. Import of entries is a sync operation that reconciles the state of the third-party system with the Data Catalog. `ImportEntries` accepts source data snapshots of a third-party system. Snapshot should be delivered as a .wire or base65-encoded .txt file containing a sequence of Protocol Buffer messages of DumpItem type. `ImportEntries` returns a long-running operation resource that can be queried with Operations.GetOperation to return ImportEntriesMetadata and an ImportEntriesResponse message.

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsEntriesImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Import')
        return self._RunMethod(config, request, global_params=global_params)
    Import.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups/{entryGroupsId}/entries:import', http_method='POST', method_id='datacatalog.projects.locations.entryGroups.entries.import', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/entries:import', request_field='googleCloudDatacatalogV1ImportEntriesRequest', request_type_name='DatacatalogProjectsLocationsEntryGroupsEntriesImportRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists entries. Note: Currently, this method can list only custom entries. To get a list of both custom and automatically created entries, use SearchCatalog.

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsEntriesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1ListEntriesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups/{entryGroupsId}/entries', http_method='GET', method_id='datacatalog.projects.locations.entryGroups.entries.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/entries', request_field='', request_type_name='DatacatalogProjectsLocationsEntryGroupsEntriesListRequest', response_type_name='GoogleCloudDatacatalogV1ListEntriesResponse', supports_download=False)

    def ModifyEntryContacts(self, request, global_params=None):
        """Modifies contacts, part of the business context of an Entry. To call this method, you must have the `datacatalog.entries.updateContacts` IAM permission on the corresponding project.

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsEntriesModifyEntryContactsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1Contacts) The response message.
      """
        config = self.GetMethodConfig('ModifyEntryContacts')
        return self._RunMethod(config, request, global_params=global_params)
    ModifyEntryContacts.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups/{entryGroupsId}/entries/{entriesId}:modifyEntryContacts', http_method='POST', method_id='datacatalog.projects.locations.entryGroups.entries.modifyEntryContacts', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:modifyEntryContacts', request_field='googleCloudDatacatalogV1ModifyEntryContactsRequest', request_type_name='DatacatalogProjectsLocationsEntryGroupsEntriesModifyEntryContactsRequest', response_type_name='GoogleCloudDatacatalogV1Contacts', supports_download=False)

    def ModifyEntryOverview(self, request, global_params=None):
        """Modifies entry overview, part of the business context of an Entry. To call this method, you must have the `datacatalog.entries.updateOverview` IAM permission on the corresponding project.

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsEntriesModifyEntryOverviewRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1EntryOverview) The response message.
      """
        config = self.GetMethodConfig('ModifyEntryOverview')
        return self._RunMethod(config, request, global_params=global_params)
    ModifyEntryOverview.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups/{entryGroupsId}/entries/{entriesId}:modifyEntryOverview', http_method='POST', method_id='datacatalog.projects.locations.entryGroups.entries.modifyEntryOverview', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:modifyEntryOverview', request_field='googleCloudDatacatalogV1ModifyEntryOverviewRequest', request_type_name='DatacatalogProjectsLocationsEntryGroupsEntriesModifyEntryOverviewRequest', response_type_name='GoogleCloudDatacatalogV1EntryOverview', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing entry. You must enable the Data Catalog API in the project identified by the `entry.name` parameter. For more information, see [Data Catalog resource project](https://cloud.google.com/data-catalog/docs/concepts/resource-project).

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsEntriesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1Entry) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups/{entryGroupsId}/entries/{entriesId}', http_method='PATCH', method_id='datacatalog.projects.locations.entryGroups.entries.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudDatacatalogV1Entry', request_type_name='DatacatalogProjectsLocationsEntryGroupsEntriesPatchRequest', response_type_name='GoogleCloudDatacatalogV1Entry', supports_download=False)

    def Star(self, request, global_params=None):
        """Marks an Entry as starred by the current user. Starring information is private to each user.

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsEntriesStarRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1StarEntryResponse) The response message.
      """
        config = self.GetMethodConfig('Star')
        return self._RunMethod(config, request, global_params=global_params)
    Star.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups/{entryGroupsId}/entries/{entriesId}:star', http_method='POST', method_id='datacatalog.projects.locations.entryGroups.entries.star', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:star', request_field='googleCloudDatacatalogV1StarEntryRequest', request_type_name='DatacatalogProjectsLocationsEntryGroupsEntriesStarRequest', response_type_name='GoogleCloudDatacatalogV1StarEntryResponse', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Gets your permissions on a resource. Returns an empty set of permissions if the resource doesn't exist. Supported resources are: - Tag templates - Entry groups Note: This method gets policies only within Data Catalog and can't be used to get policies from BigQuery, Pub/Sub, Dataproc Metastore, and any external Google Cloud Platform resources ingested into Data Catalog. No Google IAM permissions are required to call this method.

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsEntriesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups/{entryGroupsId}/entries/{entriesId}:testIamPermissions', http_method='POST', method_id='datacatalog.projects.locations.entryGroups.entries.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='DatacatalogProjectsLocationsEntryGroupsEntriesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def Unstar(self, request, global_params=None):
        """Marks an Entry as NOT starred by the current user. Starring information is private to each user.

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsEntriesUnstarRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1UnstarEntryResponse) The response message.
      """
        config = self.GetMethodConfig('Unstar')
        return self._RunMethod(config, request, global_params=global_params)
    Unstar.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups/{entryGroupsId}/entries/{entriesId}:unstar', http_method='POST', method_id='datacatalog.projects.locations.entryGroups.entries.unstar', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:unstar', request_field='googleCloudDatacatalogV1UnstarEntryRequest', request_type_name='DatacatalogProjectsLocationsEntryGroupsEntriesUnstarRequest', response_type_name='GoogleCloudDatacatalogV1UnstarEntryResponse', supports_download=False)