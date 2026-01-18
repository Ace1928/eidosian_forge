from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryResourcesFeaturesRenameRequest(_messages.Message):
    """A DirectoryResourcesFeaturesRenameRequest object.

  Fields:
    customer: The unique ID for the customer's G Suite account. As an account
      administrator, you can also use the my_customer alias to represent your
      account's customer ID.
    featureRename: A FeatureRename resource to be passed as the request body.
    oldName: The unique ID of the feature to rename.
  """
    customer = _messages.StringField(1, required=True)
    featureRename = _messages.MessageField('FeatureRename', 2)
    oldName = _messages.StringField(3, required=True)