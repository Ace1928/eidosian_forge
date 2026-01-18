from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeTargetInstancesGetRequest(_messages.Message):
    """A ComputeTargetInstancesGetRequest object.

  Fields:
    project: Project ID for this request.
    targetInstance: Name of the TargetInstance resource to return.
    zone: Name of the zone scoping this request.
  """
    project = _messages.StringField(1, required=True)
    targetInstance = _messages.StringField(2, required=True)
    zone = _messages.StringField(3, required=True)