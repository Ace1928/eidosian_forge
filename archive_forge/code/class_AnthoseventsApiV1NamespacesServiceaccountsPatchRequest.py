from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsApiV1NamespacesServiceaccountsPatchRequest(_messages.Message):
    """A AnthoseventsApiV1NamespacesServiceaccountsPatchRequest object.

  Fields:
    name: Required. The name of the serviceAccount being updated. If needed,
      replace {namespace_id} with the project ID.
    serviceAccount: A ServiceAccount resource to be passed as the request
      body.
  """
    name = _messages.StringField(1, required=True)
    serviceAccount = _messages.MessageField('ServiceAccount', 2)