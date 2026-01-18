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
class ProjectsSubscriptionsService(base_api.BaseApiService):
    """Service class for the projects_subscriptions resource."""
    _NAME = u'projects_subscriptions'

    def __init__(self, client):
        super(PubsubV1.ProjectsSubscriptionsService, self).__init__(client)
        self._upload_configs = {}

    def Acknowledge(self, request, global_params=None):
        """Acknowledges the messages associated with the `ack_ids` in the.
`AcknowledgeRequest`. The Pub/Sub system can remove the relevant messages
from the subscription.

Acknowledging a message whose ack deadline has expired may succeed,
but such a message may be redelivered later. Acknowledging a message more
than once will not result in an error.

      Args:
        request: (PubsubProjectsSubscriptionsAcknowledgeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Acknowledge')
        return self._RunMethod(config, request, global_params=global_params)
    Acknowledge.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/subscriptions/{subscriptionsId}:acknowledge', http_method=u'POST', method_id=u'pubsub.projects.subscriptions.acknowledge', ordered_params=[u'subscription'], path_params=[u'subscription'], query_params=[], relative_path=u'v1/{+subscription}:acknowledge', request_field=u'acknowledgeRequest', request_type_name=u'PubsubProjectsSubscriptionsAcknowledgeRequest', response_type_name=u'Empty', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a subscription to a given topic.
If the subscription already exists, returns `ALREADY_EXISTS`.
If the corresponding topic doesn't exist, returns `NOT_FOUND`.

If the name is not provided in the request, the server will assign a random
name for this subscription on the same project as the topic, conforming
to the
[resource name format](https://cloud.google.com/pubsub/docs/overview#names).
The generated name is populated in the returned Subscription object.
Note that for REST API requests, you must specify a name in the request.

      Args:
        request: (Subscription) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Subscription) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/subscriptions/{subscriptionsId}', http_method=u'PUT', method_id=u'pubsub.projects.subscriptions.create', ordered_params=[u'name'], path_params=[u'name'], query_params=[], relative_path=u'v1/{+name}', request_field='<request>', request_type_name=u'Subscription', response_type_name=u'Subscription', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an existing subscription. All messages retained in the subscription.
are immediately dropped. Calls to `Pull` after deletion will return
`NOT_FOUND`. After a subscription is deleted, a new one may be created with
the same name, but the new one has no association with the old
subscription or its topic unless the same topic is specified.

      Args:
        request: (PubsubProjectsSubscriptionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/subscriptions/{subscriptionsId}', http_method=u'DELETE', method_id=u'pubsub.projects.subscriptions.delete', ordered_params=[u'subscription'], path_params=[u'subscription'], query_params=[], relative_path=u'v1/{+subscription}', request_field='', request_type_name=u'PubsubProjectsSubscriptionsDeleteRequest', response_type_name=u'Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the configuration details of a subscription.

      Args:
        request: (PubsubProjectsSubscriptionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Subscription) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/subscriptions/{subscriptionsId}', http_method=u'GET', method_id=u'pubsub.projects.subscriptions.get', ordered_params=[u'subscription'], path_params=[u'subscription'], query_params=[], relative_path=u'v1/{+subscription}', request_field='', request_type_name=u'PubsubProjectsSubscriptionsGetRequest', response_type_name=u'Subscription', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource.
Returns an empty policy if the resource exists and does not have a policy
set.

      Args:
        request: (PubsubProjectsSubscriptionsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/subscriptions/{subscriptionsId}:getIamPolicy', http_method=u'GET', method_id=u'pubsub.projects.subscriptions.getIamPolicy', ordered_params=[u'resource'], path_params=[u'resource'], query_params=[], relative_path=u'v1/{+resource}:getIamPolicy', request_field='', request_type_name=u'PubsubProjectsSubscriptionsGetIamPolicyRequest', response_type_name=u'Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists matching subscriptions.

      Args:
        request: (PubsubProjectsSubscriptionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSubscriptionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/subscriptions', http_method=u'GET', method_id=u'pubsub.projects.subscriptions.list', ordered_params=[u'project'], path_params=[u'project'], query_params=[u'pageSize', u'pageToken'], relative_path=u'v1/{+project}/subscriptions', request_field='', request_type_name=u'PubsubProjectsSubscriptionsListRequest', response_type_name=u'ListSubscriptionsResponse', supports_download=False)

    def ModifyAckDeadline(self, request, global_params=None):
        """Modifies the ack deadline for a specific message. This method is useful.
to indicate that more time is needed to process a message by the
subscriber, or to make the message available for redelivery if the
processing was interrupted. Note that this does not modify the
subscription-level `ackDeadlineSeconds` used for subsequent messages.

      Args:
        request: (PubsubProjectsSubscriptionsModifyAckDeadlineRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('ModifyAckDeadline')
        return self._RunMethod(config, request, global_params=global_params)
    ModifyAckDeadline.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/subscriptions/{subscriptionsId}:modifyAckDeadline', http_method=u'POST', method_id=u'pubsub.projects.subscriptions.modifyAckDeadline', ordered_params=[u'subscription'], path_params=[u'subscription'], query_params=[], relative_path=u'v1/{+subscription}:modifyAckDeadline', request_field=u'modifyAckDeadlineRequest', request_type_name=u'PubsubProjectsSubscriptionsModifyAckDeadlineRequest', response_type_name=u'Empty', supports_download=False)

    def ModifyPushConfig(self, request, global_params=None):
        """Modifies the `PushConfig` for a specified subscription.

This may be used to change a push subscription to a pull one (signified by
an empty `PushConfig`) or vice versa, or change the endpoint URL and other
attributes of a push subscription. Messages will accumulate for delivery
continuously through the call regardless of changes to the `PushConfig`.

      Args:
        request: (PubsubProjectsSubscriptionsModifyPushConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('ModifyPushConfig')
        return self._RunMethod(config, request, global_params=global_params)
    ModifyPushConfig.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/subscriptions/{subscriptionsId}:modifyPushConfig', http_method=u'POST', method_id=u'pubsub.projects.subscriptions.modifyPushConfig', ordered_params=[u'subscription'], path_params=[u'subscription'], query_params=[], relative_path=u'v1/{+subscription}:modifyPushConfig', request_field=u'modifyPushConfigRequest', request_type_name=u'PubsubProjectsSubscriptionsModifyPushConfigRequest', response_type_name=u'Empty', supports_download=False)

    def Pull(self, request, global_params=None):
        """Pulls messages from the server. Returns an empty list if there are no.
messages available in the backlog. The server may return `UNAVAILABLE` if
there are too many concurrent pull requests pending for the given
subscription.

      Args:
        request: (PubsubProjectsSubscriptionsPullRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PullResponse) The response message.
      """
        config = self.GetMethodConfig('Pull')
        return self._RunMethod(config, request, global_params=global_params)
    Pull.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/subscriptions/{subscriptionsId}:pull', http_method=u'POST', method_id=u'pubsub.projects.subscriptions.pull', ordered_params=[u'subscription'], path_params=[u'subscription'], query_params=[], relative_path=u'v1/{+subscription}:pull', request_field=u'pullRequest', request_type_name=u'PubsubProjectsSubscriptionsPullRequest', response_type_name=u'PullResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any.
existing policy.

      Args:
        request: (PubsubProjectsSubscriptionsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/subscriptions/{subscriptionsId}:setIamPolicy', http_method=u'POST', method_id=u'pubsub.projects.subscriptions.setIamPolicy', ordered_params=[u'resource'], path_params=[u'resource'], query_params=[], relative_path=u'v1/{+resource}:setIamPolicy', request_field=u'setIamPolicyRequest', request_type_name=u'PubsubProjectsSubscriptionsSetIamPolicyRequest', response_type_name=u'Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.
If the resource does not exist, this will return an empty set of
permissions, not a NOT_FOUND error.

Note: This operation is designed to be used for building permission-aware
UIs and command-line tools, not for authorization checking. This operation
may "fail open" without warning.

      Args:
        request: (PubsubProjectsSubscriptionsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/subscriptions/{subscriptionsId}:testIamPermissions', http_method=u'POST', method_id=u'pubsub.projects.subscriptions.testIamPermissions', ordered_params=[u'resource'], path_params=[u'resource'], query_params=[], relative_path=u'v1/{+resource}:testIamPermissions', request_field=u'testIamPermissionsRequest', request_type_name=u'PubsubProjectsSubscriptionsTestIamPermissionsRequest', response_type_name=u'TestIamPermissionsResponse', supports_download=False)