from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.policytroubleshooter.v3alpha import policytroubleshooter_v3alpha_messages as messages
class GcpUserAccessBindingService(base_api.BaseApiService):
    """Service class for the gcpUserAccessBinding resource."""
    _NAME = 'gcpUserAccessBinding'

    def __init__(self, client):
        super(PolicytroubleshooterV3alpha.GcpUserAccessBindingService, self).__init__(client)
        self._upload_configs = {}

    def Troubleshoot(self, request, global_params=None):
        """Checks why an access is granted or not with GcpUserAccessBinding.

      Args:
        request: (GoogleCloudPolicytroubleshooterGcpuseraccessbindingV3alphaTroubleshootGcpUserAccessBindingRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudPolicytroubleshooterGcpuseraccessbindingV3alphaTroubleshootGcpUserAccessBindingResponse) The response message.
      """
        config = self.GetMethodConfig('Troubleshoot')
        return self._RunMethod(config, request, global_params=global_params)
    Troubleshoot.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='policytroubleshooter.gcpUserAccessBinding.troubleshoot', ordered_params=[], path_params=[], query_params=[], relative_path='v3alpha/gcpUserAccessBinding:troubleshoot', request_field='<request>', request_type_name='GoogleCloudPolicytroubleshooterGcpuseraccessbindingV3alphaTroubleshootGcpUserAccessBindingRequest', response_type_name='GoogleCloudPolicytroubleshooterGcpuseraccessbindingV3alphaTroubleshootGcpUserAccessBindingResponse', supports_download=False)