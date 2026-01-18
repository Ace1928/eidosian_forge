from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsApiproductsService(base_api.BaseApiService):
    """Service class for the organizations_apiproducts resource."""
    _NAME = 'organizations_apiproducts'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsApiproductsService, self).__init__(client)
        self._upload_configs = {}

    def Attributes(self, request, global_params=None):
        """Updates or creates API product attributes. This API **replaces** the current list of attributes with the attributes specified in the request body. In this way, you can update existing attributes, add new attributes, or delete existing attributes by omitting them from the request body. **Note**: OAuth access tokens and Key Management Service (KMS) entities (apps, developers, and API products) are cached for 180 seconds (current default). Any custom attributes associated with entities also get cached for at least 180 seconds after entity is accessed during runtime. In this case, the `ExpiresIn` element on the OAuthV2 policy won't be able to expire an access token in less than 180 seconds.

      Args:
        request: (ApigeeOrganizationsApiproductsAttributesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Attributes) The response message.
      """
        config = self.GetMethodConfig('Attributes')
        return self._RunMethod(config, request, global_params=global_params)
    Attributes.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apiproducts/{apiproductsId}/attributes', http_method='POST', method_id='apigee.organizations.apiproducts.attributes', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}/attributes', request_field='googleCloudApigeeV1Attributes', request_type_name='ApigeeOrganizationsApiproductsAttributesRequest', response_type_name='GoogleCloudApigeeV1Attributes', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates an API product in an organization. You create API products after you have proxied backend services using API proxies. An API product is a collection of API resources combined with quota settings and metadata that you can use to deliver customized and productized API bundles to your developer community. This metadata can include: - Scope - Environments - API proxies - Extensible profile API products enable you repackage APIs on the fly, without having to do any additional coding or configuration. Apigee recommends that you start with a simple API product including only required elements. You then provision credentials to apps to enable them to start testing your APIs. After you have authentication and authorization working against a simple API product, you can iterate to create finer-grained API products, defining different sets of API resources for each API product. **WARNING:** - If you don't specify an API proxy in the request body, *any* app associated with the product can make calls to *any* API in your entire organization. - If you don't specify an environment in the request body, the product allows access to all environments. For more information, see What is an API product?.

      Args:
        request: (ApigeeOrganizationsApiproductsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiProduct) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apiproducts', http_method='POST', method_id='apigee.organizations.apiproducts.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/apiproducts', request_field='googleCloudApigeeV1ApiProduct', request_type_name='ApigeeOrganizationsApiproductsCreateRequest', response_type_name='GoogleCloudApigeeV1ApiProduct', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an API product from an organization. Deleting an API product causes app requests to the resource URIs defined in the API product to fail. Ensure that you create a new API product to serve existing apps, unless your intention is to disable access to the resources defined in the API product. The API product name required in the request URL is the internal name of the product, not the display name. While they may be the same, it depends on whether the API product was created via the UI or the API. View the list of API products to verify the internal name.

      Args:
        request: (ApigeeOrganizationsApiproductsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiProduct) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apiproducts/{apiproductsId}', http_method='DELETE', method_id='apigee.organizations.apiproducts.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsApiproductsDeleteRequest', response_type_name='GoogleCloudApigeeV1ApiProduct', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets configuration details for an API product. The API product name required in the request URL is the internal name of the product, not the display name. While they may be the same, it depends on whether the API product was created via the UI or the API. View the list of API products to verify the internal name.

      Args:
        request: (ApigeeOrganizationsApiproductsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiProduct) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apiproducts/{apiproductsId}', http_method='GET', method_id='apigee.organizations.apiproducts.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsApiproductsGetRequest', response_type_name='GoogleCloudApigeeV1ApiProduct', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all API product names for an organization. Filter the list by passing an `attributename` and `attibutevalue`. The maximum number of API products returned is 1000. You can paginate the list of API products returned using the `startKey` and `count` query parameters.

      Args:
        request: (ApigeeOrganizationsApiproductsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListApiProductsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apiproducts', http_method='GET', method_id='apigee.organizations.apiproducts.list', ordered_params=['parent'], path_params=['parent'], query_params=['attributename', 'attributevalue', 'count', 'expand', 'filter', 'pageSize', 'pageToken', 'startKey'], relative_path='v1/{+parent}/apiproducts', request_field='', request_type_name='ApigeeOrganizationsApiproductsListRequest', response_type_name='GoogleCloudApigeeV1ListApiProductsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates an existing API product. You must include all required values, whether or not you are updating them, as well as any optional values that you are updating. The API product name required in the request URL is the internal name of the product, not the display name. While they may be the same, it depends on whether the API product was created via UI or API. View the list of API products to identify their internal names.

      Args:
        request: (GoogleCloudApigeeV1ApiProduct) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiProduct) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apiproducts/{apiproductsId}', http_method='PUT', method_id='apigee.organizations.apiproducts.update', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='<request>', request_type_name='GoogleCloudApigeeV1ApiProduct', response_type_name='GoogleCloudApigeeV1ApiProduct', supports_download=False)