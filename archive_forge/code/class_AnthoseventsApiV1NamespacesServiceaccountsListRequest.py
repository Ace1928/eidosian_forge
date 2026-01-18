from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsApiV1NamespacesServiceaccountsListRequest(_messages.Message):
    """A AnthoseventsApiV1NamespacesServiceaccountsListRequest object.

  Fields:
    parent: The project ID or project number from which the service account
      should be listed.
  """
    parent = _messages.StringField(1, required=True)