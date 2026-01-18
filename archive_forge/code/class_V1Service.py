from __future__ import absolute_import
from apitools.base.py import base_api
from samples.servicemanagement_sample.servicemanagement_v1 import servicemanagement_v1_messages as messages
from the newest to the oldest.
class V1Service(base_api.BaseApiService):
    """Service class for the v1 resource."""
    _NAME = u'v1'

    def __init__(self, client):
        super(ServicemanagementV1.V1Service, self).__init__(client)
        self._upload_configs = {}

    def ConvertConfig(self, request, global_params=None):
        """DEPRECATED. `SubmitConfigSource` with `validate_only=true` will provide.
config conversion moving forward.

Converts an API specification (e.g. Swagger spec) to an
equivalent `google.api.Service`.

      Args:
        request: (ConvertConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConvertConfigResponse) The response message.
      """
        config = self.GetMethodConfig('ConvertConfig')
        return self._RunMethod(config, request, global_params=global_params)
    ConvertConfig.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'servicemanagement.convertConfig', ordered_params=[], path_params=[], query_params=[], relative_path=u'v1:convertConfig', request_field='<request>', request_type_name=u'ConvertConfigRequest', response_type_name=u'ConvertConfigResponse', supports_download=False)