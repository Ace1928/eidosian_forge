from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentmanagerDeploymentsDeleteRequest(_messages.Message):
    """A DeploymentmanagerDeploymentsDeleteRequest object.

  Enums:
    DeletePolicyValueValuesEnum: Sets the policy to use for deleting
      resources.

  Fields:
    deletePolicy: Sets the policy to use for deleting resources.
    deployment: The name of the deployment for this request.
    project: The project ID for this request.
  """

    class DeletePolicyValueValuesEnum(_messages.Enum):
        """Sets the policy to use for deleting resources.

    Values:
      DELETE: <no description>
      ABANDON: <no description>
    """
        DELETE = 0
        ABANDON = 1
    deletePolicy = _messages.EnumField('DeletePolicyValueValuesEnum', 1, default='DELETE')
    deployment = _messages.StringField(2, required=True)
    project = _messages.StringField(3, required=True)