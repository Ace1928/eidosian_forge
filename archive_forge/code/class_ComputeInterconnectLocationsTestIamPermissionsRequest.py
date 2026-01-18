from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeInterconnectLocationsTestIamPermissionsRequest(_messages.Message):
    """A ComputeInterconnectLocationsTestIamPermissionsRequest object.

  Fields:
    project: Project ID for this request.
    resource: Name or id of the resource for this request.
    testPermissionsRequest: A TestPermissionsRequest resource to be passed as
      the request body.
  """
    project = _messages.StringField(1, required=True)
    resource = _messages.StringField(2, required=True)
    testPermissionsRequest = _messages.MessageField('TestPermissionsRequest', 3)