from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vision.v1 import vision_v1_messages as messages
class ProjectsLocationsProductSetsService(base_api.BaseApiService):
    """Service class for the projects_locations_productSets resource."""
    _NAME = 'projects_locations_productSets'

    def __init__(self, client):
        super(VisionV1.ProjectsLocationsProductSetsService, self).__init__(client)
        self._upload_configs = {}

    def AddProduct(self, request, global_params=None):
        """Adds a Product to the specified ProductSet. If the Product is already present, no change is made. One Product can be added to at most 100 ProductSets. Possible errors: * Returns NOT_FOUND if the Product or the ProductSet doesn't exist.

      Args:
        request: (VisionProjectsLocationsProductSetsAddProductRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('AddProduct')
        return self._RunMethod(config, request, global_params=global_params)
    AddProduct.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/productSets/{productSetsId}:addProduct', http_method='POST', method_id='vision.projects.locations.productSets.addProduct', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:addProduct', request_field='addProductToProductSetRequest', request_type_name='VisionProjectsLocationsProductSetsAddProductRequest', response_type_name='Empty', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates and returns a new ProductSet resource. Possible errors: * Returns INVALID_ARGUMENT if display_name is missing, or is longer than 4096 characters.

      Args:
        request: (VisionProjectsLocationsProductSetsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ProductSet) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/productSets', http_method='POST', method_id='vision.projects.locations.productSets.create', ordered_params=['parent'], path_params=['parent'], query_params=['productSetId'], relative_path='v1/{+parent}/productSets', request_field='productSet', request_type_name='VisionProjectsLocationsProductSetsCreateRequest', response_type_name='ProductSet', supports_download=False)

    def Delete(self, request, global_params=None):
        """Permanently deletes a ProductSet. Products and ReferenceImages in the ProductSet are not deleted. The actual image files are not deleted from Google Cloud Storage.

      Args:
        request: (VisionProjectsLocationsProductSetsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/productSets/{productSetsId}', http_method='DELETE', method_id='vision.projects.locations.productSets.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VisionProjectsLocationsProductSetsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information associated with a ProductSet. Possible errors: * Returns NOT_FOUND if the ProductSet does not exist.

      Args:
        request: (VisionProjectsLocationsProductSetsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ProductSet) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/productSets/{productSetsId}', http_method='GET', method_id='vision.projects.locations.productSets.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VisionProjectsLocationsProductSetsGetRequest', response_type_name='ProductSet', supports_download=False)

    def Import(self, request, global_params=None):
        """Asynchronous API that imports a list of reference images to specified product sets based on a list of image information. The google.longrunning.Operation API can be used to keep track of the progress and results of the request. `Operation.metadata` contains `BatchOperationMetadata`. (progress) `Operation.response` contains `ImportProductSetsResponse`. (results) The input source of this method is a csv file on Google Cloud Storage. For the format of the csv file please see ImportProductSetsGcsSource.csv_file_uri.

      Args:
        request: (VisionProjectsLocationsProductSetsImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Import')
        return self._RunMethod(config, request, global_params=global_params)
    Import.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/productSets:import', http_method='POST', method_id='vision.projects.locations.productSets.import', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/productSets:import', request_field='importProductSetsRequest', request_type_name='VisionProjectsLocationsProductSetsImportRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ProductSets in an unspecified order. Possible errors: * Returns INVALID_ARGUMENT if page_size is greater than 100, or less than 1.

      Args:
        request: (VisionProjectsLocationsProductSetsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListProductSetsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/productSets', http_method='GET', method_id='vision.projects.locations.productSets.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/productSets', request_field='', request_type_name='VisionProjectsLocationsProductSetsListRequest', response_type_name='ListProductSetsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Makes changes to a ProductSet resource. Only display_name can be updated currently. Possible errors: * Returns NOT_FOUND if the ProductSet does not exist. * Returns INVALID_ARGUMENT if display_name is present in update_mask but missing from the request or longer than 4096 characters.

      Args:
        request: (VisionProjectsLocationsProductSetsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ProductSet) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/productSets/{productSetsId}', http_method='PATCH', method_id='vision.projects.locations.productSets.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='productSet', request_type_name='VisionProjectsLocationsProductSetsPatchRequest', response_type_name='ProductSet', supports_download=False)

    def RemoveProduct(self, request, global_params=None):
        """Removes a Product from the specified ProductSet.

      Args:
        request: (VisionProjectsLocationsProductSetsRemoveProductRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('RemoveProduct')
        return self._RunMethod(config, request, global_params=global_params)
    RemoveProduct.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/productSets/{productSetsId}:removeProduct', http_method='POST', method_id='vision.projects.locations.productSets.removeProduct', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:removeProduct', request_field='removeProductFromProductSetRequest', request_type_name='VisionProjectsLocationsProductSetsRemoveProductRequest', response_type_name='Empty', supports_download=False)