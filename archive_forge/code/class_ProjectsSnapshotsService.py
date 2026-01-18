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
class ProjectsSnapshotsService(base_api.BaseApiService):
    """Service class for the projects_snapshots resource."""
    _NAME = u'projects_snapshots'

    def __init__(self, client):
        super(PubsubV1.ProjectsSnapshotsService, self).__init__(client)
        self._upload_configs = {}

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource.
Returns an empty policy if the resource exists and does not have a policy
set.

      Args:
        request: (PubsubProjectsSnapshotsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/snapshots/{snapshotsId}:getIamPolicy', http_method=u'GET', method_id=u'pubsub.projects.snapshots.getIamPolicy', ordered_params=[u'resource'], path_params=[u'resource'], query_params=[], relative_path=u'v1/{+resource}:getIamPolicy', request_field='', request_type_name=u'PubsubProjectsSnapshotsGetIamPolicyRequest', response_type_name=u'Policy', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any.
existing policy.

      Args:
        request: (PubsubProjectsSnapshotsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/snapshots/{snapshotsId}:setIamPolicy', http_method=u'POST', method_id=u'pubsub.projects.snapshots.setIamPolicy', ordered_params=[u'resource'], path_params=[u'resource'], query_params=[], relative_path=u'v1/{+resource}:setIamPolicy', request_field=u'setIamPolicyRequest', request_type_name=u'PubsubProjectsSnapshotsSetIamPolicyRequest', response_type_name=u'Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.
If the resource does not exist, this will return an empty set of
permissions, not a NOT_FOUND error.

Note: This operation is designed to be used for building permission-aware
UIs and command-line tools, not for authorization checking. This operation
may "fail open" without warning.

      Args:
        request: (PubsubProjectsSnapshotsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path=u'v1/projects/{projectsId}/snapshots/{snapshotsId}:testIamPermissions', http_method=u'POST', method_id=u'pubsub.projects.snapshots.testIamPermissions', ordered_params=[u'resource'], path_params=[u'resource'], query_params=[], relative_path=u'v1/{+resource}:testIamPermissions', request_field=u'testIamPermissionsRequest', request_type_name=u'PubsubProjectsSnapshotsTestIamPermissionsRequest', response_type_name=u'TestIamPermissionsResponse', supports_download=False)