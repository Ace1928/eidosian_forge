from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageBasicCardButtonOpenUriAction(_messages.Message):
    """Opens the given URI.

  Fields:
    uri: Required. The HTTP or HTTPS scheme URI.
  """
    uri = _messages.StringField(1)