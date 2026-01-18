from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsApiV1NamespacesServiceaccountsCreateRequest(_messages.Message):
    """A AnthoseventsApiV1NamespacesServiceaccountsCreateRequest object.

  Fields:
    parent: Required. The namespace in which this service account should be
      created.
    serviceAccount: A ServiceAccount resource to be passed as the request
      body.
  """
    parent = _messages.StringField(1, required=True)
    serviceAccount = _messages.MessageField('ServiceAccount', 2)