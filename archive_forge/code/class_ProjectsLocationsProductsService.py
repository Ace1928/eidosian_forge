from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vision.v1 import vision_v1_messages as messages
class ProjectsLocationsProductsService(base_api.BaseApiService):
    """Service class for the projects_locations_products resource."""
    _NAME = 'projects_locations_products'

    def __init__(self, client):
        super(VisionV1.ProjectsLocationsProductsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates and returns a new product resource. Possible errors: * Returns INVALID_ARGUMENT if display_name is missing or longer than 4096 characters. * Returns INVALID_ARGUMENT if description is longer than 4096 characters. * Returns INVALID_ARGUMENT if product_category is missing or invalid.

      Args:
        request: (VisionProjectsLocationsProductsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Product) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/products', http_method='POST', method_id='vision.projects.locations.products.create', ordered_params=['parent'], path_params=['parent'], query_params=['productId'], relative_path='v1/{+parent}/products', request_field='product', request_type_name='VisionProjectsLocationsProductsCreateRequest', response_type_name='Product', supports_download=False)

    def Delete(self, request, global_params=None):
        """Permanently deletes a product and its reference images. Metadata of the product and all its images will be deleted right away, but search queries against ProductSets containing the product may still work until all related caches are refreshed.

      Args:
        request: (VisionProjectsLocationsProductsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/products/{productsId}', http_method='DELETE', method_id='vision.projects.locations.products.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VisionProjectsLocationsProductsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information associated with a Product. Possible errors: * Returns NOT_FOUND if the Product does not exist.

      Args:
        request: (VisionProjectsLocationsProductsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Product) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/products/{productsId}', http_method='GET', method_id='vision.projects.locations.products.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VisionProjectsLocationsProductsGetRequest', response_type_name='Product', supports_download=False)

    def List(self, request, global_params=None):
        """Lists products in an unspecified order. Possible errors: * Returns INVALID_ARGUMENT if page_size is greater than 100 or less than 1.

      Args:
        request: (VisionProjectsLocationsProductsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListProductsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/products', http_method='GET', method_id='vision.projects.locations.products.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/products', request_field='', request_type_name='VisionProjectsLocationsProductsListRequest', response_type_name='ListProductsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Makes changes to a Product resource. Only the `display_name`, `description`, and `labels` fields can be updated right now. If labels are updated, the change will not be reflected in queries until the next index time. Possible errors: * Returns NOT_FOUND if the Product does not exist. * Returns INVALID_ARGUMENT if display_name is present in update_mask but is missing from the request or longer than 4096 characters. * Returns INVALID_ARGUMENT if description is present in update_mask but is longer than 4096 characters. * Returns INVALID_ARGUMENT if product_category is present in update_mask.

      Args:
        request: (VisionProjectsLocationsProductsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Product) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/products/{productsId}', http_method='PATCH', method_id='vision.projects.locations.products.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='product', request_type_name='VisionProjectsLocationsProductsPatchRequest', response_type_name='Product', supports_download=False)

    def Purge(self, request, global_params=None):
        """Asynchronous API to delete all Products in a ProductSet or all Products that are in no ProductSet. If a Product is a member of the specified ProductSet in addition to other ProductSets, the Product will still be deleted. It is recommended to not delete the specified ProductSet until after this operation has completed. It is also recommended to not add any of the Products involved in the batch delete to a new ProductSet while this operation is running because those Products may still end up deleted. It's not possible to undo the PurgeProducts operation. Therefore, it is recommended to keep the csv files used in ImportProductSets (if that was how you originally built the Product Set) before starting PurgeProducts, in case you need to re-import the data after deletion. If the plan is to purge all of the Products from a ProductSet and then re-use the empty ProductSet to re-import new Products into the empty ProductSet, you must wait until the PurgeProducts operation has finished for that ProductSet. The google.longrunning.Operation API can be used to keep track of the progress and results of the request. `Operation.metadata` contains `BatchOperationMetadata`. (progress).

      Args:
        request: (VisionProjectsLocationsProductsPurgeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Purge')
        return self._RunMethod(config, request, global_params=global_params)
    Purge.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/products:purge', http_method='POST', method_id='vision.projects.locations.products.purge', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/products:purge', request_field='purgeProductsRequest', request_type_name='VisionProjectsLocationsProductsPurgeRequest', response_type_name='Operation', supports_download=False)