from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vision.v1 import vision_v1_messages as messages
class ProjectsLocationsProductsReferenceImagesService(base_api.BaseApiService):
    """Service class for the projects_locations_products_referenceImages resource."""
    _NAME = 'projects_locations_products_referenceImages'

    def __init__(self, client):
        super(VisionV1.ProjectsLocationsProductsReferenceImagesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates and returns a new ReferenceImage resource. The `bounding_poly` field is optional. If `bounding_poly` is not specified, the system will try to detect regions of interest in the image that are compatible with the product_category on the parent product. If it is specified, detection is ALWAYS skipped. The system converts polygons into non-rotated rectangles. Note that the pipeline will resize the image if the image resolution is too large to process (above 50MP). Possible errors: * Returns INVALID_ARGUMENT if the image_uri is missing or longer than 4096 characters. * Returns INVALID_ARGUMENT if the product does not exist. * Returns INVALID_ARGUMENT if bounding_poly is not provided, and nothing compatible with the parent product's product_category is detected. * Returns INVALID_ARGUMENT if bounding_poly contains more than 10 polygons.

      Args:
        request: (VisionProjectsLocationsProductsReferenceImagesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReferenceImage) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/products/{productsId}/referenceImages', http_method='POST', method_id='vision.projects.locations.products.referenceImages.create', ordered_params=['parent'], path_params=['parent'], query_params=['referenceImageId'], relative_path='v1/{+parent}/referenceImages', request_field='referenceImage', request_type_name='VisionProjectsLocationsProductsReferenceImagesCreateRequest', response_type_name='ReferenceImage', supports_download=False)

    def Delete(self, request, global_params=None):
        """Permanently deletes a reference image. The image metadata will be deleted right away, but search queries against ProductSets containing the image may still work until all related caches are refreshed. The actual image files are not deleted from Google Cloud Storage.

      Args:
        request: (VisionProjectsLocationsProductsReferenceImagesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/products/{productsId}/referenceImages/{referenceImagesId}', http_method='DELETE', method_id='vision.projects.locations.products.referenceImages.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VisionProjectsLocationsProductsReferenceImagesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information associated with a ReferenceImage. Possible errors: * Returns NOT_FOUND if the specified image does not exist.

      Args:
        request: (VisionProjectsLocationsProductsReferenceImagesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReferenceImage) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/products/{productsId}/referenceImages/{referenceImagesId}', http_method='GET', method_id='vision.projects.locations.products.referenceImages.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VisionProjectsLocationsProductsReferenceImagesGetRequest', response_type_name='ReferenceImage', supports_download=False)

    def List(self, request, global_params=None):
        """Lists reference images. Possible errors: * Returns NOT_FOUND if the parent product does not exist. * Returns INVALID_ARGUMENT if the page_size is greater than 100, or less than 1.

      Args:
        request: (VisionProjectsLocationsProductsReferenceImagesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListReferenceImagesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/products/{productsId}/referenceImages', http_method='GET', method_id='vision.projects.locations.products.referenceImages.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/referenceImages', request_field='', request_type_name='VisionProjectsLocationsProductsReferenceImagesListRequest', response_type_name='ListReferenceImagesResponse', supports_download=False)