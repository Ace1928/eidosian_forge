from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datacatalog.v1 import datacatalog_v1_messages as messages
class ProjectsLocationsEntryGroupsService(base_api.BaseApiService):
    """Service class for the projects_locations_entryGroups resource."""
    _NAME = 'projects_locations_entryGroups'

    def __init__(self, client):
        super(DatacatalogV1.ProjectsLocationsEntryGroupsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an entry group. An entry group contains logically related entries together with [Cloud Identity and Access Management](/data-catalog/docs/concepts/iam) policies. These policies specify users who can create, edit, and view entries within entry groups. Data Catalog automatically creates entry groups with names that start with the `@` symbol for the following resources: * BigQuery entries (`@bigquery`) * Pub/Sub topics (`@pubsub`) * Dataproc Metastore services (`@dataproc_metastore_{SERVICE_NAME_HASH}`) You can create your own entry groups for Cloud Storage fileset entries and custom entries together with the corresponding IAM policies. User-created entry groups can't contain the `@` symbol, it is reserved for automatically created groups. Entry groups, like entries, can be searched. A maximum of 10,000 entry groups may be created per organization across all locations. You must enable the Data Catalog API in the project identified by the `parent` parameter. For more information, see [Data Catalog resource project](https://cloud.google.com/data-catalog/docs/concepts/resource-project).

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1EntryGroup) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups', http_method='POST', method_id='datacatalog.projects.locations.entryGroups.create', ordered_params=['parent'], path_params=['parent'], query_params=['entryGroupId'], relative_path='v1/{+parent}/entryGroups', request_field='googleCloudDatacatalogV1EntryGroup', request_type_name='DatacatalogProjectsLocationsEntryGroupsCreateRequest', response_type_name='GoogleCloudDatacatalogV1EntryGroup', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an entry group. You must enable the Data Catalog API in the project identified by the `name` parameter. For more information, see [Data Catalog resource project](https://cloud.google.com/data-catalog/docs/concepts/resource-project).

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups/{entryGroupsId}', http_method='DELETE', method_id='datacatalog.projects.locations.entryGroups.delete', ordered_params=['name'], path_params=['name'], query_params=['force'], relative_path='v1/{+name}', request_field='', request_type_name='DatacatalogProjectsLocationsEntryGroupsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an entry group.

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1EntryGroup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups/{entryGroupsId}', http_method='GET', method_id='datacatalog.projects.locations.entryGroups.get', ordered_params=['name'], path_params=['name'], query_params=['readMask'], relative_path='v1/{+name}', request_field='', request_type_name='DatacatalogProjectsLocationsEntryGroupsGetRequest', response_type_name='GoogleCloudDatacatalogV1EntryGroup', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May return: * A`NOT_FOUND` error if the resource doesn't exist or you don't have the permission to view it. * An empty policy if the resource exists but doesn't have a set policy. Supported resources are: - Tag templates - Entry groups Note: This method doesn't get policies from Google Cloud Platform resources ingested into Data Catalog. To call this method, you must have the following Google IAM permissions: - `datacatalog.tagTemplates.getIamPolicy` to get policies on tag templates. - `datacatalog.entryGroups.getIamPolicy` to get policies on entry groups.

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups/{entryGroupsId}:getIamPolicy', http_method='POST', method_id='datacatalog.projects.locations.entryGroups.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='DatacatalogProjectsLocationsEntryGroupsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists entry groups.

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1ListEntryGroupsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups', http_method='GET', method_id='datacatalog.projects.locations.entryGroups.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/entryGroups', request_field='', request_type_name='DatacatalogProjectsLocationsEntryGroupsListRequest', response_type_name='GoogleCloudDatacatalogV1ListEntryGroupsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an entry group. You must enable the Data Catalog API in the project identified by the `entry_group.name` parameter. For more information, see [Data Catalog resource project](https://cloud.google.com/data-catalog/docs/concepts/resource-project).

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1EntryGroup) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups/{entryGroupsId}', http_method='PATCH', method_id='datacatalog.projects.locations.entryGroups.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudDatacatalogV1EntryGroup', request_type_name='DatacatalogProjectsLocationsEntryGroupsPatchRequest', response_type_name='GoogleCloudDatacatalogV1EntryGroup', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets an access control policy for a resource. Replaces any existing policy. Supported resources are: - Tag templates - Entry groups Note: This method sets policies only within Data Catalog and can't be used to manage policies in BigQuery, Pub/Sub, Dataproc Metastore, and any external Google Cloud Platform resources synced with the Data Catalog. To call this method, you must have the following Google IAM permissions: - `datacatalog.tagTemplates.setIamPolicy` to set policies on tag templates. - `datacatalog.entryGroups.setIamPolicy` to set policies on entry groups.

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups/{entryGroupsId}:setIamPolicy', http_method='POST', method_id='datacatalog.projects.locations.entryGroups.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='DatacatalogProjectsLocationsEntryGroupsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Gets your permissions on a resource. Returns an empty set of permissions if the resource doesn't exist. Supported resources are: - Tag templates - Entry groups Note: This method gets policies only within Data Catalog and can't be used to get policies from BigQuery, Pub/Sub, Dataproc Metastore, and any external Google Cloud Platform resources ingested into Data Catalog. No Google IAM permissions are required to call this method.

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/entryGroups/{entryGroupsId}:testIamPermissions', http_method='POST', method_id='datacatalog.projects.locations.entryGroups.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='DatacatalogProjectsLocationsEntryGroupsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)