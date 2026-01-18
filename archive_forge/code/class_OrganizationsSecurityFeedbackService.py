from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsSecurityFeedbackService(base_api.BaseApiService):
    """Service class for the organizations_securityFeedback resource."""
    _NAME = 'organizations_securityFeedback'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsSecurityFeedbackService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new report containing customer feedback.

      Args:
        request: (ApigeeOrganizationsSecurityFeedbackCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityFeedback) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/securityFeedback', http_method='POST', method_id='apigee.organizations.securityFeedback.create', ordered_params=['parent'], path_params=['parent'], query_params=['securityFeedbackId'], relative_path='v1/{+parent}/securityFeedback', request_field='googleCloudApigeeV1SecurityFeedback', request_type_name='ApigeeOrganizationsSecurityFeedbackCreateRequest', response_type_name='GoogleCloudApigeeV1SecurityFeedback', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a specific feedback report. Used for "undo" of a feedback submission.

      Args:
        request: (ApigeeOrganizationsSecurityFeedbackDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/securityFeedback/{securityFeedbackId}', http_method='DELETE', method_id='apigee.organizations.securityFeedback.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsSecurityFeedbackDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a specific customer feedback report.

      Args:
        request: (ApigeeOrganizationsSecurityFeedbackGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityFeedback) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/securityFeedback/{securityFeedbackId}', http_method='GET', method_id='apigee.organizations.securityFeedback.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsSecurityFeedbackGetRequest', response_type_name='GoogleCloudApigeeV1SecurityFeedback', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all feedback reports which have already been submitted.

      Args:
        request: (ApigeeOrganizationsSecurityFeedbackListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListSecurityFeedbackResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/securityFeedback', http_method='GET', method_id='apigee.organizations.securityFeedback.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/securityFeedback', request_field='', request_type_name='ApigeeOrganizationsSecurityFeedbackListRequest', response_type_name='GoogleCloudApigeeV1ListSecurityFeedbackResponse', supports_download=False)