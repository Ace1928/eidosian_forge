from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV3betaListPolicyBindingsResponse(_messages.Message):
    """Response message for ListPolicyBindings method.

  Fields:
    nextPageToken: Optional. A token, which can be sent as `page_token` to
      retrieve the next page. If this field is omitted, there are no
      subsequent pages.
    policyBindings: The policy bindings from the specified parent.
  """
    nextPageToken = _messages.StringField(1)
    policyBindings = _messages.MessageField('GoogleIamV3betaPolicyBinding', 2, repeated=True)