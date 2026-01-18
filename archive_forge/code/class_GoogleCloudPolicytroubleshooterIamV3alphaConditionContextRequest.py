from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3alphaConditionContextRequest(_messages.Message):
    """This message defines attributes for an HTTP request. If the actual
  request is not an HTTP request, the runtime system should try to map the
  actual request to an equivalent HTTP request.

  Fields:
    receiveTime: Optional. The timestamp when the destination service receives
      the first byte of the request.
    satisfiedAccessLevels: Optional. The information for access levels that
      are satisfied for the given access tuple.
    unsatisfiedAccessLevels: Optional. The information for access levels that
      are unsatisfied for the given access tuple.
  """
    receiveTime = _messages.StringField(1)
    satisfiedAccessLevels = _messages.StringField(2, repeated=True)
    unsatisfiedAccessLevels = _messages.StringField(3, repeated=True)