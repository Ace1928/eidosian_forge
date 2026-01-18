from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sqladmin.v1beta4 import sqladmin_v1beta4_messages as messages
class ConnectService(base_api.BaseApiService):
    """Service class for the connect resource."""
    _NAME = 'connect'

    def __init__(self, client):
        super(SqladminV1beta4.ConnectService, self).__init__(client)
        self._upload_configs = {}

    def GenerateEphemeralCert(self, request, global_params=None):
        """Generates a short-lived X509 certificate containing the provided public key and signed by a private key specific to the target instance. Users may use the certificate to authenticate as themselves when connecting to the database.

      Args:
        request: (SqlConnectGenerateEphemeralRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GenerateEphemeralCertResponse) The response message.
      """
        config = self.GetMethodConfig('GenerateEphemeralCert')
        return self._RunMethod(config, request, global_params=global_params)
    GenerateEphemeralCert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.connect.generateEphemeral', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}:generateEphemeralCert', request_field='generateEphemeralCertRequest', request_type_name='SqlConnectGenerateEphemeralRequest', response_type_name='GenerateEphemeralCertResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves connect settings about a Cloud SQL instance.

      Args:
        request: (SqlConnectGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConnectSettings) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='sql.connect.get', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=['readTime'], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/connectSettings', request_field='', request_type_name='SqlConnectGetRequest', response_type_name='ConnectSettings', supports_download=False)