from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsDevelopersAttributesService(base_api.BaseApiService):
    """Service class for the organizations_developers_attributes resource."""
    _NAME = 'organizations_developers_attributes'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsDevelopersAttributesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a developer attribute.

      Args:
        request: (ApigeeOrganizationsDevelopersAttributesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Attribute) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/attributes/{attributesId}', http_method='DELETE', method_id='apigee.organizations.developers.attributes.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsDevelopersAttributesDeleteRequest', response_type_name='GoogleCloudApigeeV1Attribute', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the value of the specified developer attribute.

      Args:
        request: (ApigeeOrganizationsDevelopersAttributesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Attribute) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/attributes/{attributesId}', http_method='GET', method_id='apigee.organizations.developers.attributes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsDevelopersAttributesGetRequest', response_type_name='GoogleCloudApigeeV1Attribute', supports_download=False)

    def List(self, request, global_params=None):
        """Returns a list of all developer attributes.

      Args:
        request: (ApigeeOrganizationsDevelopersAttributesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Attributes) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/attributes', http_method='GET', method_id='apigee.organizations.developers.attributes.list', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/attributes', request_field='', request_type_name='ApigeeOrganizationsDevelopersAttributesListRequest', response_type_name='GoogleCloudApigeeV1Attributes', supports_download=False)

    def UpdateDeveloperAttribute(self, request, global_params=None):
        """Updates a developer attribute. **Note**: OAuth access tokens and Key Management Service (KMS) entities (apps, developers, and API products) are cached for 180 seconds (default). Any custom attributes associated with these entities are cached for at least 180 seconds after the entity is accessed at runtime. Therefore, an `ExpiresIn` element on the OAuthV2 policy won't be able to expire an access token in less than 180 seconds.

      Args:
        request: (GoogleCloudApigeeV1Attribute) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Attribute) The response message.
      """
        config = self.GetMethodConfig('UpdateDeveloperAttribute')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateDeveloperAttribute.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/attributes/{attributesId}', http_method='POST', method_id='apigee.organizations.developers.attributes.updateDeveloperAttribute', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='<request>', request_type_name='GoogleCloudApigeeV1Attribute', response_type_name='GoogleCloudApigeeV1Attribute', supports_download=False)