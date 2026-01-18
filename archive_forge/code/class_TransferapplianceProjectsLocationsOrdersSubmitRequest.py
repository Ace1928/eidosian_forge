from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferapplianceProjectsLocationsOrdersSubmitRequest(_messages.Message):
    """A TransferapplianceProjectsLocationsOrdersSubmitRequest object.

  Fields:
    name: Required. Name of the Order resource to submit.
    submitOrderRequest: A SubmitOrderRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    submitOrderRequest = _messages.MessageField('SubmitOrderRequest', 2)