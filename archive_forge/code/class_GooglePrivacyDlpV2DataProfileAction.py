from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DataProfileAction(_messages.Message):
    """A task to execute when a data profile has been generated.

  Fields:
    exportData: Export data profiles into a provided location.
    pubSubNotification: Publish a message into the Pub/Sub topic.
  """
    exportData = _messages.MessageField('GooglePrivacyDlpV2Export', 1)
    pubSubNotification = _messages.MessageField('GooglePrivacyDlpV2PubSubNotification', 2)