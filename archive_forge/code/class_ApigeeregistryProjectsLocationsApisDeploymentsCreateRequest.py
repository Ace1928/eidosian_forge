from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsApisDeploymentsCreateRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsApisDeploymentsCreateRequest object.

  Fields:
    apiDeployment: A ApiDeployment resource to be passed as the request body.
    apiDeploymentId: Required. The ID to use for the deployment, which will
      become the final component of the deployment's resource name. This value
      should be 4-63 characters, and valid characters are /a-z-/. Following
      AIP-162, IDs must not have the form of a UUID.
    parent: Required. The parent, which owns this collection of deployments.
      Format: `projects/*/locations/*/apis/*`
  """
    apiDeployment = _messages.MessageField('ApiDeployment', 1)
    apiDeploymentId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)