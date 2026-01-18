from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamLocationsWorkforcePoolsPatchRequest(_messages.Message):
    """A IamLocationsWorkforcePoolsPatchRequest object.

  Fields:
    name: Output only. The resource name of the pool. Format:
      `locations/{location}/workforcePools/{workforce_pool_id}`
    updateMask: Required. The list of fields to update.
    workforcePool: A WorkforcePool resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    workforcePool = _messages.MessageField('WorkforcePool', 3)