from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FeedOutputConfig(_messages.Message):
    """Output configuration for asset feed destination.

  Fields:
    pubsubDestination: Destination on Pub/Sub.
  """
    pubsubDestination = _messages.MessageField('PubsubDestination', 1)