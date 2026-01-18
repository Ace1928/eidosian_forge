from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalCustomer(_messages.Message):
    """Entity representing a SAS customer.

  Fields:
    displayName: Required. Name of the organization that the customer entity
      represents.
    name: Output only. Resource name of the customer.
    sasUserIds: User IDs used by the devices belonging to this customer.
  """
    displayName = _messages.StringField(1)
    name = _messages.StringField(2)
    sasUserIds = _messages.StringField(3, repeated=True)