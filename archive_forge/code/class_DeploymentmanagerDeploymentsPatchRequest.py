from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentmanagerDeploymentsPatchRequest(_messages.Message):
    """A DeploymentmanagerDeploymentsPatchRequest object.

  Enums:
    CreatePolicyValueValuesEnum: Sets the policy to use for creating new
      resources.
    DeletePolicyValueValuesEnum: Sets the policy to use for deleting
      resources.

  Fields:
    createPolicy: Sets the policy to use for creating new resources.
    deletePolicy: Sets the policy to use for deleting resources.
    deployment: The name of the deployment for this request.
    deploymentResource: A Deployment resource to be passed as the request
      body.
    preview: If set to true, updates the deployment and creates and updates
      the "shell" resources but does not actually alter or instantiate these
      resources. This allows you to preview what your deployment will look
      like. You can use this intent to preview how an update would affect your
      deployment. You must provide a `target.config` with a configuration if
      this is set to true. After previewing a deployment, you can deploy your
      resources by making a request with the `update()` or you can
      `cancelPreview()` to remove the preview altogether. Note that the
      deployment will still exist after you cancel the preview and you must
      separately delete this deployment if you want to remove it.
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

    class DeletePolicyValueValuesEnum(_messages.Enum):
        """Sets the policy to use for deleting resources.

    Values:
      DELETE: <no description>
      ABANDON: <no description>
    """
        DELETE = 0
        ABANDON = 1
    createPolicy = _messages.EnumField('CreatePolicyValueValuesEnum', 1, default='CREATE_OR_ACQUIRE')
    deletePolicy = _messages.EnumField('DeletePolicyValueValuesEnum', 2, default='DELETE')
    deployment = _messages.StringField(3, required=True)
    deploymentResource = _messages.MessageField('Deployment', 4)
    preview = _messages.BooleanField(5, default=False)
    project = _messages.StringField(6, required=True)