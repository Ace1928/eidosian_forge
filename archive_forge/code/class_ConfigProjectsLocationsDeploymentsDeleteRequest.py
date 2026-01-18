from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigProjectsLocationsDeploymentsDeleteRequest(_messages.Message):
    """A ConfigProjectsLocationsDeploymentsDeleteRequest object.

  Enums:
    DeletePolicyValueValuesEnum: Optional. Policy on how resources actuated by
      the deployment should be deleted. If unspecified, the default behavior
      is to delete the underlying resources.

  Fields:
    deletePolicy: Optional. Policy on how resources actuated by the deployment
      should be deleted. If unspecified, the default behavior is to delete the
      underlying resources.
    force: Optional. If set to true, any revisions for this deployment will
      also be deleted. (Otherwise, the request will only work if the
      deployment has no revisions.)
    name: Required. The name of the Deployment in the format:
      'projects/{project_id}/locations/{location}/deployments/{deployment}'.
    requestId: Optional. An optional request ID to identify requests. Specify
      a unique request ID so that if you must retry your request, the server
      will know to ignore the request if it has already been completed. The
      server will guarantee that for at least 60 minutes after the first
      request. For example, consider a situation where you make an initial
      request and the request times out. If you make the request again with
      the same request ID, the server can check if original operation with the
      same request ID was received, and if so, will ignore the second request.
      This prevents clients from accidentally creating duplicate commitments.
      The request ID must be a valid UUID with the exception that zero UUID is
      not supported (00000000-0000-0000-0000-000000000000).
  """

    class DeletePolicyValueValuesEnum(_messages.Enum):
        """Optional. Policy on how resources actuated by the deployment should be
    deleted. If unspecified, the default behavior is to delete the underlying
    resources.

    Values:
      DELETE_POLICY_UNSPECIFIED: Unspecified policy, resources will be
        deleted.
      DELETE: Deletes resources actuated by the deployment.
      ABANDON: Abandons resources and only deletes the deployment and its
        metadata.
    """
        DELETE_POLICY_UNSPECIFIED = 0
        DELETE = 1
        ABANDON = 2
    deletePolicy = _messages.EnumField('DeletePolicyValueValuesEnum', 1)
    force = _messages.BooleanField(2)
    name = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)