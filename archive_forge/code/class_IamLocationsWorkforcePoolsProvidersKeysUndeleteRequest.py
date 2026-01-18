from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamLocationsWorkforcePoolsProvidersKeysUndeleteRequest(_messages.Message):
    """A IamLocationsWorkforcePoolsProvidersKeysUndeleteRequest object.

  Fields:
    name: Required. The name of the key to undelete.
    undeleteWorkforcePoolProviderKeyRequest: A
      UndeleteWorkforcePoolProviderKeyRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    undeleteWorkforcePoolProviderKeyRequest = _messages.MessageField('UndeleteWorkforcePoolProviderKeyRequest', 2)