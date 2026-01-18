from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.policysimulator.v1beta import policysimulator_v1beta_messages as messages
class OrganizationsLocationsOrgPolicyViolationsPreviewsOrgPolicyViolationsService(base_api.BaseApiService):
    """Service class for the organizations_locations_orgPolicyViolationsPreviews_orgPolicyViolations resource."""
    _NAME = 'organizations_locations_orgPolicyViolationsPreviews_orgPolicyViolations'

    def __init__(self, client):
        super(PolicysimulatorV1beta.OrganizationsLocationsOrgPolicyViolationsPreviewsOrgPolicyViolationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """GetOrgPolicyViolation gets the specified OrgPolicyViolation that is present in an OrgPolicyViolationsPreview. This method is currently unimplemented.

      Args:
        request: (PolicysimulatorOrganizationsLocationsOrgPolicyViolationsPreviewsOrgPolicyViolationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudPolicysimulatorV1betaOrgPolicyViolation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/organizations/{organizationsId}/locations/{locationsId}/orgPolicyViolationsPreviews/{orgPolicyViolationsPreviewsId}/orgPolicyViolations/{orgPolicyViolationsId}', http_method='GET', method_id='policysimulator.organizations.locations.orgPolicyViolationsPreviews.orgPolicyViolations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='PolicysimulatorOrganizationsLocationsOrgPolicyViolationsPreviewsOrgPolicyViolationsGetRequest', response_type_name='GoogleCloudPolicysimulatorV1betaOrgPolicyViolation', supports_download=False)

    def List(self, request, global_params=None):
        """ListOrgPolicyViolations lists the OrgPolicyViolations that are present in an OrgPolicyViolationsPreview.

      Args:
        request: (PolicysimulatorOrganizationsLocationsOrgPolicyViolationsPreviewsOrgPolicyViolationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudPolicysimulatorV1betaListOrgPolicyViolationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/organizations/{organizationsId}/locations/{locationsId}/orgPolicyViolationsPreviews/{orgPolicyViolationsPreviewsId}/orgPolicyViolations', http_method='GET', method_id='policysimulator.organizations.locations.orgPolicyViolationsPreviews.orgPolicyViolations.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/orgPolicyViolations', request_field='', request_type_name='PolicysimulatorOrganizationsLocationsOrgPolicyViolationsPreviewsOrgPolicyViolationsListRequest', response_type_name='GoogleCloudPolicysimulatorV1betaListOrgPolicyViolationsResponse', supports_download=False)