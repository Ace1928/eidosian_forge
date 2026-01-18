from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v3 import monitoring_v3_messages as messages
class ProjectsNotificationChannelsService(base_api.BaseApiService):
    """Service class for the projects_notificationChannels resource."""
    _NAME = 'projects_notificationChannels'

    def __init__(self, client):
        super(MonitoringV3.ProjectsNotificationChannelsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new notification channel, representing a single notification endpoint such as an email address, SMS number, or PagerDuty service.Design your application to single-thread API calls that modify the state of notification channels in a single project. This includes calls to CreateNotificationChannel, DeleteNotificationChannel and UpdateNotificationChannel.

      Args:
        request: (MonitoringProjectsNotificationChannelsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NotificationChannel) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/notificationChannels', http_method='POST', method_id='monitoring.projects.notificationChannels.create', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}/notificationChannels', request_field='notificationChannel', request_type_name='MonitoringProjectsNotificationChannelsCreateRequest', response_type_name='NotificationChannel', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a notification channel.Design your application to single-thread API calls that modify the state of notification channels in a single project. This includes calls to CreateNotificationChannel, DeleteNotificationChannel and UpdateNotificationChannel.

      Args:
        request: (MonitoringProjectsNotificationChannelsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/notificationChannels/{notificationChannelsId}', http_method='DELETE', method_id='monitoring.projects.notificationChannels.delete', ordered_params=['name'], path_params=['name'], query_params=['force'], relative_path='v3/{+name}', request_field='', request_type_name='MonitoringProjectsNotificationChannelsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a single notification channel. The channel includes the relevant configuration details with which the channel was created. However, the response may truncate or omit passwords, API keys, or other private key matter and thus the response may not be 100% identical to the information that was supplied in the call to the create method.

      Args:
        request: (MonitoringProjectsNotificationChannelsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NotificationChannel) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/notificationChannels/{notificationChannelsId}', http_method='GET', method_id='monitoring.projects.notificationChannels.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}', request_field='', request_type_name='MonitoringProjectsNotificationChannelsGetRequest', response_type_name='NotificationChannel', supports_download=False)

    def GetVerificationCode(self, request, global_params=None):
        """Requests a verification code for an already verified channel that can then be used in a call to VerifyNotificationChannel() on a different channel with an equivalent identity in the same or in a different project. This makes it possible to copy a channel between projects without requiring manual reverification of the channel. If the channel is not in the verified state, this method will fail (in other words, this may only be used if the SendNotificationChannelVerificationCode and VerifyNotificationChannel paths have already been used to put the given channel into the verified state).There is no guarantee that the verification codes returned by this method will be of a similar structure or form as the ones that are delivered to the channel via SendNotificationChannelVerificationCode; while VerifyNotificationChannel() will recognize both the codes delivered via SendNotificationChannelVerificationCode() and returned from GetNotificationChannelVerificationCode(), it is typically the case that the verification codes delivered via SendNotificationChannelVerificationCode() will be shorter and also have a shorter expiration (e.g. codes such as "G-123456") whereas GetVerificationCode() will typically return a much longer, websafe base 64 encoded string that has a longer expiration time.

      Args:
        request: (MonitoringProjectsNotificationChannelsGetVerificationCodeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GetNotificationChannelVerificationCodeResponse) The response message.
      """
        config = self.GetMethodConfig('GetVerificationCode')
        return self._RunMethod(config, request, global_params=global_params)
    GetVerificationCode.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/notificationChannels/{notificationChannelsId}:getVerificationCode', http_method='POST', method_id='monitoring.projects.notificationChannels.getVerificationCode', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}:getVerificationCode', request_field='getNotificationChannelVerificationCodeRequest', request_type_name='MonitoringProjectsNotificationChannelsGetVerificationCodeRequest', response_type_name='GetNotificationChannelVerificationCodeResponse', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the notification channels that have been created for the project. To list the types of notification channels that are supported, use the ListNotificationChannelDescriptors method.

      Args:
        request: (MonitoringProjectsNotificationChannelsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNotificationChannelsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/notificationChannels', http_method='GET', method_id='monitoring.projects.notificationChannels.list', ordered_params=['name'], path_params=['name'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v3/{+name}/notificationChannels', request_field='', request_type_name='MonitoringProjectsNotificationChannelsListRequest', response_type_name='ListNotificationChannelsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a notification channel. Fields not specified in the field mask remain unchanged.Design your application to single-thread API calls that modify the state of notification channels in a single project. This includes calls to CreateNotificationChannel, DeleteNotificationChannel and UpdateNotificationChannel.

      Args:
        request: (MonitoringProjectsNotificationChannelsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NotificationChannel) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/notificationChannels/{notificationChannelsId}', http_method='PATCH', method_id='monitoring.projects.notificationChannels.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v3/{+name}', request_field='notificationChannel', request_type_name='MonitoringProjectsNotificationChannelsPatchRequest', response_type_name='NotificationChannel', supports_download=False)

    def SendVerificationCode(self, request, global_params=None):
        """Causes a verification code to be delivered to the channel. The code can then be supplied in VerifyNotificationChannel to verify the channel.

      Args:
        request: (MonitoringProjectsNotificationChannelsSendVerificationCodeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('SendVerificationCode')
        return self._RunMethod(config, request, global_params=global_params)
    SendVerificationCode.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/notificationChannels/{notificationChannelsId}:sendVerificationCode', http_method='POST', method_id='monitoring.projects.notificationChannels.sendVerificationCode', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}:sendVerificationCode', request_field='sendNotificationChannelVerificationCodeRequest', request_type_name='MonitoringProjectsNotificationChannelsSendVerificationCodeRequest', response_type_name='Empty', supports_download=False)

    def Verify(self, request, global_params=None):
        """Verifies a NotificationChannel by proving receipt of the code delivered to the channel as a result of calling SendNotificationChannelVerificationCode.

      Args:
        request: (MonitoringProjectsNotificationChannelsVerifyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NotificationChannel) The response message.
      """
        config = self.GetMethodConfig('Verify')
        return self._RunMethod(config, request, global_params=global_params)
    Verify.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/notificationChannels/{notificationChannelsId}:verify', http_method='POST', method_id='monitoring.projects.notificationChannels.verify', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}:verify', request_field='verifyNotificationChannelRequest', request_type_name='MonitoringProjectsNotificationChannelsVerifyRequest', response_type_name='NotificationChannel', supports_download=False)