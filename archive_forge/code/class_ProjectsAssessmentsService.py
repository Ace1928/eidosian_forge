from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recaptchaenterprise.v1 import recaptchaenterprise_v1_messages as messages
class ProjectsAssessmentsService(base_api.BaseApiService):
    """Service class for the projects_assessments resource."""
    _NAME = 'projects_assessments'

    def __init__(self, client):
        super(RecaptchaenterpriseV1.ProjectsAssessmentsService, self).__init__(client)
        self._upload_configs = {}

    def Annotate(self, request, global_params=None):
        """Annotates a previously created Assessment to provide additional information on whether the event turned out to be authentic or fraudulent.

      Args:
        request: (RecaptchaenterpriseProjectsAssessmentsAnnotateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1AnnotateAssessmentResponse) The response message.
      """
        config = self.GetMethodConfig('Annotate')
        return self._RunMethod(config, request, global_params=global_params)
    Annotate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/assessments/{assessmentsId}:annotate', http_method='POST', method_id='recaptchaenterprise.projects.assessments.annotate', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:annotate', request_field='googleCloudRecaptchaenterpriseV1AnnotateAssessmentRequest', request_type_name='RecaptchaenterpriseProjectsAssessmentsAnnotateRequest', response_type_name='GoogleCloudRecaptchaenterpriseV1AnnotateAssessmentResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates an Assessment of the likelihood an event is legitimate.

      Args:
        request: (RecaptchaenterpriseProjectsAssessmentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1Assessment) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/assessments', http_method='POST', method_id='recaptchaenterprise.projects.assessments.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/assessments', request_field='googleCloudRecaptchaenterpriseV1Assessment', request_type_name='RecaptchaenterpriseProjectsAssessmentsCreateRequest', response_type_name='GoogleCloudRecaptchaenterpriseV1Assessment', supports_download=False)