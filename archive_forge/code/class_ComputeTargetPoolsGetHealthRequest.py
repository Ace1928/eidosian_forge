from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeTargetPoolsGetHealthRequest(_messages.Message):
    """A ComputeTargetPoolsGetHealthRequest object.

  Fields:
    instanceReference: A InstanceReference resource to be passed as the
      request body.
    project: Project ID for this request.
    region: Name of the region scoping this request.
    targetPool: Name of the TargetPool resource to which the queried instance
      belongs.
  """
    instanceReference = _messages.MessageField('InstanceReference', 1)
    project = _messages.StringField(2, required=True)
    region = _messages.StringField(3, required=True)
    targetPool = _messages.StringField(4, required=True)