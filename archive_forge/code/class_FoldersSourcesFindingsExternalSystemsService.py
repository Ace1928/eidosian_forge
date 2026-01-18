from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v2 import securitycenter_v2_messages as messages
class FoldersSourcesFindingsExternalSystemsService(base_api.BaseApiService):
    """Service class for the folders_sources_findings_externalSystems resource."""
    _NAME = 'folders_sources_findings_externalSystems'

    def __init__(self, client):
        super(SecuritycenterV2.FoldersSourcesFindingsExternalSystemsService, self).__init__(client)
        self._upload_configs = {}

    def Patch(self, request, global_params=None):
        """Updates external system. This is for a given finding. If no location is specified, finding is assumed to be in global.

      Args:
        request: (SecuritycenterFoldersSourcesFindingsExternalSystemsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV2ExternalSystem) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/folders/{foldersId}/sources/{sourcesId}/findings/{findingsId}/externalSystems/{externalSystemsId}', http_method='PATCH', method_id='securitycenter.folders.sources.findings.externalSystems.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='googleCloudSecuritycenterV2ExternalSystem', request_type_name='SecuritycenterFoldersSourcesFindingsExternalSystemsPatchRequest', response_type_name='GoogleCloudSecuritycenterV2ExternalSystem', supports_download=False)