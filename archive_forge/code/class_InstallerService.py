from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sasportal.v1alpha1 import sasportal_v1alpha1_messages as messages
class InstallerService(base_api.BaseApiService):
    """Service class for the installer resource."""
    _NAME = 'installer'

    def __init__(self, client):
        super(SasportalV1alpha1.InstallerService, self).__init__(client)
        self._upload_configs = {}

    def GenerateSecret(self, request, global_params=None):
        """Generates a secret to be used with the ValidateInstaller.

      Args:
        request: (SasPortalGenerateSecretRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalGenerateSecretResponse) The response message.
      """
        config = self.GetMethodConfig('GenerateSecret')
        return self._RunMethod(config, request, global_params=global_params)
    GenerateSecret.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sasportal.installer.generateSecret', ordered_params=[], path_params=[], query_params=[], relative_path='v1alpha1/installer:generateSecret', request_field='<request>', request_type_name='SasPortalGenerateSecretRequest', response_type_name='SasPortalGenerateSecretResponse', supports_download=False)

    def Validate(self, request, global_params=None):
        """Validates the identity of a Certified Professional Installer (CPI).

      Args:
        request: (SasPortalValidateInstallerRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalValidateInstallerResponse) The response message.
      """
        config = self.GetMethodConfig('Validate')
        return self._RunMethod(config, request, global_params=global_params)
    Validate.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sasportal.installer.validate', ordered_params=[], path_params=[], query_params=[], relative_path='v1alpha1/installer:validate', request_field='<request>', request_type_name='SasPortalValidateInstallerRequest', response_type_name='SasPortalValidateInstallerResponse', supports_download=False)