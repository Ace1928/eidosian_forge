from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentmanagerDeploymentsInsertRequest(_messages.Message):
    """A DeploymentmanagerDeploymentsInsertRequest object.

  Enums:
    CreatePolicyValueValuesEnum: Sets the policy to use for creating new
      resources.

  Fields:
    createPolicy: Sets the policy to use for creating new resources.
    deployment: A Deployment resource to be passed as the request body.
    preview: If set to true, creates a deployment and creates "shell"
      resources but does not actually instantiate these resources. This allows
      you to preview what your deployment looks like. After previewing a
      deployment, you can deploy your resources by making a request with the
      `update()` method or you can use the `cancelPreview()` method to cancel
      the preview altogether. Note that the deployment will still exist after
      you cancel the preview and you must separately delete this deployment if
      you want to remove it.
    project: The project ID for this request.
  """

    class CreatePolicyValueValuesEnum(_messages.Enum):
        """Sets the policy to use for creating new resources.

    Values:
      CREATE_OR_ACQUIRE: <no description>
      ACQUIRE: <no description>
    """
        CREATE_OR_ACQUIRE = 0
        ACQUIRE = 1
    createPolicy = _messages.EnumField('CreatePolicyValueValuesEnum', 1, default='CREATE_OR_ACQUIRE')
    deployment = _messages.MessageField('Deployment', 2)
    preview = _messages.BooleanField(3)
    project = _messages.StringField(4, required=True)