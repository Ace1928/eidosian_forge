from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryResourcesFeaturesDeleteRequest(_messages.Message):
    """A DirectoryResourcesFeaturesDeleteRequest object.

  Fields:
    customer: The unique ID for the customer's G Suite account. As an account
      administrator, you can also use the my_customer alias to represent your
      account's customer ID.
    featureKey: The unique ID of the feature to delete.
  """
    customer = _messages.StringField(1, required=True)
    featureKey = _messages.StringField(2, required=True)