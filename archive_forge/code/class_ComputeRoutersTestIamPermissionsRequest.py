from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeRoutersTestIamPermissionsRequest(_messages.Message):
    """A ComputeRoutersTestIamPermissionsRequest object.

  Fields:
    project: Project ID for this request.
    region: The name of the region for this request.
    resource: Name or id of the resource for this request.
    testPermissionsRequest: A TestPermissionsRequest resource to be passed as
      the request body.
  """
    project = _messages.StringField(1, required=True)
    region = _messages.StringField(2, required=True)
    resource = _messages.StringField(3, required=True)
    testPermissionsRequest = _messages.MessageField('TestPermissionsRequest', 4)