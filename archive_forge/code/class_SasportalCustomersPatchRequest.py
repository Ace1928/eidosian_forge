from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasportalCustomersPatchRequest(_messages.Message):
    """A SasportalCustomersPatchRequest object.

  Fields:
    name: Output only. Resource name of the customer.
    sasPortalCustomer: A SasPortalCustomer resource to be passed as the
      request body.
    updateMask: Fields to be updated.
  """
    name = _messages.StringField(1, required=True)
    sasPortalCustomer = _messages.MessageField('SasPortalCustomer', 2)
    updateMask = _messages.StringField(3)