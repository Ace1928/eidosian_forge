from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryResourcesFeaturesInsertRequest(_messages.Message):
    """A DirectoryResourcesFeaturesInsertRequest object.

  Fields:
    customer: The unique ID for the customer's G Suite account. As an account
      administrator, you can also use the my_customer alias to represent your
      account's customer ID.
    feature: A Feature resource to be passed as the request body.
  """
    customer = _messages.StringField(1, required=True)
    feature = _messages.MessageField('Feature', 2)