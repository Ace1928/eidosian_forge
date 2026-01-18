from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DataProfilePubSubCondition(_messages.Message):
    """A condition for determining whether a Pub/Sub should be triggered.

  Fields:
    expressions: An expression.
  """
    expressions = _messages.MessageField('GooglePrivacyDlpV2PubSubExpressions', 1)