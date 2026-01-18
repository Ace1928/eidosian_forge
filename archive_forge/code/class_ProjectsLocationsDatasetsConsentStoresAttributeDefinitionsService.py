from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1alpha2 import healthcare_v1alpha2_messages as messages
class ProjectsLocationsDatasetsConsentStoresAttributeDefinitionsService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_consentStores_attributeDefinitions resource."""
    _NAME = 'projects_locations_datasets_consentStores_attributeDefinitions'

    def __init__(self, client):
        super(HealthcareV1alpha2.ProjectsLocationsDatasetsConsentStoresAttributeDefinitionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Attribute definition in the parent consent store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresAttributeDefinitionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AttributeDefinition) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}/attributeDefinitions', http_method='POST', method_id='healthcare.projects.locations.datasets.consentStores.attributeDefinitions.create', ordered_params=['parent'], path_params=['parent'], query_params=['attributeDefinitionId'], relative_path='v1alpha2/{+parent}/attributeDefinitions', request_field='attributeDefinition', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresAttributeDefinitionsCreateRequest', response_type_name='AttributeDefinition', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified Attribute definition. Fails if the Attribute definition is referenced by any User data mapping, the latest revision of any Consent, or the latest approved revision of any Consent content.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresAttributeDefinitionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}/attributeDefinitions/{attributeDefinitionsId}', http_method='DELETE', method_id='healthcare.projects.locations.datasets.consentStores.attributeDefinitions.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresAttributeDefinitionsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the specified Attribute definition.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresAttributeDefinitionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AttributeDefinition) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}/attributeDefinitions/{attributeDefinitionsId}', http_method='GET', method_id='healthcare.projects.locations.datasets.consentStores.attributeDefinitions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresAttributeDefinitionsGetRequest', response_type_name='AttributeDefinition', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the Attribute definitions in the specified consent store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresAttributeDefinitionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAttributeDefinitionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}/attributeDefinitions', http_method='GET', method_id='healthcare.projects.locations.datasets.consentStores.attributeDefinitions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/attributeDefinitions', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresAttributeDefinitionsListRequest', response_type_name='ListAttributeDefinitionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified Attribute definition.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresAttributeDefinitionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AttributeDefinition) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}/attributeDefinitions/{attributeDefinitionsId}', http_method='PATCH', method_id='healthcare.projects.locations.datasets.consentStores.attributeDefinitions.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha2/{+name}', request_field='attributeDefinition', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresAttributeDefinitionsPatchRequest', response_type_name='AttributeDefinition', supports_download=False)