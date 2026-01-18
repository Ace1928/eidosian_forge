from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkmanagementProjectsLocationsDeploymentsCreateRequest(_messages.Message):
    """A NetworkmanagementProjectsLocationsDeploymentsCreateRequest object.

  Fields:
    deployment: A Deployment resource to be passed as the request body.
    deploymentId: The identifier for this deployment instance.
    parent: Required. The project and location where the Deployment should be
      created, specified in the format
      `projects/{project}/locations/{location}`.
  """
    deployment = _messages.MessageField('Deployment', 1)
    deploymentId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)