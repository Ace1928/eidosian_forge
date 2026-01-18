from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeInstancesGetGuestAttributesRequest(_messages.Message):
    """A ComputeInstancesGetGuestAttributesRequest object.

  Fields:
    instance: Name of the instance scoping this request.
    project: Project ID for this request.
    queryPath: Specifies the guest attributes path to be queried.
    variableKey: Specifies the key for the guest attributes entry.
    zone: The name of the zone for this request.
  """
    instance = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    queryPath = _messages.StringField(3)
    variableKey = _messages.StringField(4)
    zone = _messages.StringField(5, required=True)