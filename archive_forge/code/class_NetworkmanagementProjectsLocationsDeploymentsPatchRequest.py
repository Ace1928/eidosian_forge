from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkmanagementProjectsLocationsDeploymentsPatchRequest(_messages.Message):
    """A NetworkmanagementProjectsLocationsDeploymentsPatchRequest object.

  Fields:
    deployment: A Deployment resource to be passed as the request body.
    name: Required. Name of the Deployment resource. It matches the pattern
      `projects/{project}/locations/{location}/deployments/{deployment}` and
      must be unique.
    updateMask: update mask
  """
    deployment = _messages.MessageField('Deployment', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)