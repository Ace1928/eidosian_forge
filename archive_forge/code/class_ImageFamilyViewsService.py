from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class ImageFamilyViewsService(base_api.BaseApiService):
    """Service class for the imageFamilyViews resource."""
    _NAME = 'imageFamilyViews'

    def __init__(self, client):
        super(ComputeBeta.ImageFamilyViewsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Returns the latest image that is part of an image family, is not deprecated and is rolled out in the specified zone.

      Args:
        request: (ComputeImageFamilyViewsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ImageFamilyView) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.imageFamilyViews.get', ordered_params=['project', 'zone', 'family'], path_params=['family', 'project', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/imageFamilyViews/{family}', request_field='', request_type_name='ComputeImageFamilyViewsGetRequest', response_type_name='ImageFamilyView', supports_download=False)