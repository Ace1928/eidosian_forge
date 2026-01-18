from __future__ import absolute_import
import os
import platform
import sys
from apitools.base.py import base_api
import gslib.third_party.pubsub_apitools.pubsub_v1_messages as messages
import gslib
from gslib.metrics import MetricsCollector
from gslib.utils import system_util
from the subscription.
class ProjectsTopicsService(base_api.BaseApiService):
    """Service class for the projects_topics resource."""
    _NAME = u'projects_topics'

    def __init__(self, client):
        super(PubsubV1.ProjectsTopicsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates the given topic with the given name.

      Args:
        request: (Topic) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Topic) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/topics/{topicsId}', http_method=u'PUT', method_id=u'pubsub.projects.topics.create', ordered_params=[u'name'], path_params=[u'name'], query_params=[], relative_path=u'v1/{+name}', request_field='<request>', request_type_name=u'Topic', response_type_name=u'Topic', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the topic with the given name. Returns `NOT_FOUND` if the topic.
does not exist. After a topic is deleted, a new topic may be created with
the same name; this is an entirely new topic with none of the old
configuration or subscriptions. Existing subscriptions to this topic are
not deleted, but their `topic` field is set to `_deleted-topic_`.

      Args:
        request: (PubsubProjectsTopicsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/topics/{topicsId}', http_method=u'DELETE', method_id=u'pubsub.projects.topics.delete', ordered_params=[u'topic'], path_params=[u'topic'], query_params=[], relative_path=u'v1/{+topic}', request_field='', request_type_name=u'PubsubProjectsTopicsDeleteRequest', response_type_name=u'Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the configuration of a topic.

      Args:
        request: (PubsubProjectsTopicsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Topic) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/topics/{topicsId}', http_method=u'GET', method_id=u'pubsub.projects.topics.get', ordered_params=[u'topic'], path_params=[u'topic'], query_params=[], relative_path=u'v1/{+topic}', request_field='', request_type_name=u'PubsubProjectsTopicsGetRequest', response_type_name=u'Topic', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource.
Returns an empty policy if the resource exists and does not have a policy
set.

      Args:
        request: (PubsubProjectsTopicsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/topics/{topicsId}:getIamPolicy', http_method=u'GET', method_id=u'pubsub.projects.topics.getIamPolicy', ordered_params=[u'resource'], path_params=[u'resource'], query_params=[], relative_path=u'v1/{+resource}:getIamPolicy', request_field='', request_type_name=u'PubsubProjectsTopicsGetIamPolicyRequest', response_type_name=u'Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists matching topics.

      Args:
        request: (PubsubProjectsTopicsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTopicsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/topics', http_method=u'GET', method_id=u'pubsub.projects.topics.list', ordered_params=[u'project'], path_params=[u'project'], query_params=[u'pageSize', u'pageToken'], relative_path=u'v1/{+project}/topics', request_field='', request_type_name=u'PubsubProjectsTopicsListRequest', response_type_name=u'ListTopicsResponse', supports_download=False)

    def Publish(self, request, global_params=None):
        """Adds one or more messages to the topic. Returns `NOT_FOUND` if the topic.
does not exist. The message payload must not be empty; it must contain
 either a non-empty data field, or at least one attribute.

      Args:
        request: (PubsubProjectsTopicsPublishRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PublishResponse) The response message.
      """
        config = self.GetMethodConfig('Publish')
        return self._RunMethod(config, request, global_params=global_params)
    Publish.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/topics/{topicsId}:publish', http_method=u'POST', method_id=u'pubsub.projects.topics.publish', ordered_params=[u'topic'], path_params=[u'topic'], query_params=[], relative_path=u'v1/{+topic}:publish', request_field=u'publishRequest', request_type_name=u'PubsubProjectsTopicsPublishRequest', response_type_name=u'PublishResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any.
existing policy.

      Args:
        request: (PubsubProjectsTopicsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/topics/{topicsId}:setIamPolicy', http_method=u'POST', method_id=u'pubsub.projects.topics.setIamPolicy', ordered_params=[u'resource'], path_params=[u'resource'], query_params=[], relative_path=u'v1/{+resource}:setIamPolicy', request_field=u'setIamPolicyRequest', request_type_name=u'PubsubProjectsTopicsSetIamPolicyRequest', response_type_name=u'Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.
If the resource does not exist, this will return an empty set of
permissions, not a NOT_FOUND error.

Note: This operation is designed to be used for building permission-aware
UIs and command-line tools, not for authorization checking. This operation
may "fail open" without warning.

      Args:
        request: (PubsubProjectsTopicsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/topics/{topicsId}:testIamPermissions', http_method=u'POST', method_id=u'pubsub.projects.topics.testIamPermissions', ordered_params=[u'resource'], path_params=[u'resource'], query_params=[], relative_path=u'v1/{+resource}:testIamPermissions', request_field=u'testIamPermissionsRequest', request_type_name=u'PubsubProjectsTopicsTestIamPermissionsRequest', response_type_name=u'TestIamPermissionsResponse', supports_download=False)