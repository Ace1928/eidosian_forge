from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vision.v1 import vision_v1_messages as messages
class ImagesService(base_api.BaseApiService):
    """Service class for the images resource."""
    _NAME = 'images'

    def __init__(self, client):
        super(VisionV1.ImagesService, self).__init__(client)
        self._upload_configs = {}

    def Annotate(self, request, global_params=None):
        """Run image detection and annotation for a batch of images.

      Args:
        request: (BatchAnnotateImagesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BatchAnnotateImagesResponse) The response message.
      """
        config = self.GetMethodConfig('Annotate')
        return self._RunMethod(config, request, global_params=global_params)
    Annotate.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='vision.images.annotate', ordered_params=[], path_params=[], query_params=[], relative_path='v1/images:annotate', request_field='<request>', request_type_name='BatchAnnotateImagesRequest', response_type_name='BatchAnnotateImagesResponse', supports_download=False)

    def AsyncBatchAnnotate(self, request, global_params=None):
        """Run asynchronous image detection and annotation for a list of images. Progress and results can be retrieved through the `google.longrunning.Operations` interface. `Operation.metadata` contains `OperationMetadata` (metadata). `Operation.response` contains `AsyncBatchAnnotateImagesResponse` (results). This service will write image annotation outputs to json files in customer GCS bucket, each json file containing BatchAnnotateImagesResponse proto.

      Args:
        request: (AsyncBatchAnnotateImagesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AsyncBatchAnnotate')
        return self._RunMethod(config, request, global_params=global_params)
    AsyncBatchAnnotate.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='vision.images.asyncBatchAnnotate', ordered_params=[], path_params=[], query_params=[], relative_path='v1/images:asyncBatchAnnotate', request_field='<request>', request_type_name='AsyncBatchAnnotateImagesRequest', response_type_name='Operation', supports_download=False)