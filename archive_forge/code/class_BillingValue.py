from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BillingValue(_messages.Message):
    """The bucket's billing configuration.

    Fields:
      requesterPays: When set to true, Requester Pays is enabled for this
        bucket.
    """
    requesterPays = _messages.BooleanField(1)