from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1DeveloperSubscription(_messages.Message):
    """Structure of a DeveloperSubscription.

  Fields:
    apiproduct: Name of the API product for which the developer is purchasing
      a subscription.
    createdAt: Output only. Time when the API product subscription was created
      in milliseconds since epoch.
    endTime: Time when the API product subscription ends in milliseconds since
      epoch.
    lastModifiedAt: Output only. Time when the API product subscription was
      last modified in milliseconds since epoch.
    name: Output only. Name of the API product subscription.
    startTime: Time when the API product subscription starts in milliseconds
      since epoch.
  """
    apiproduct = _messages.StringField(1)
    createdAt = _messages.IntegerField(2)
    endTime = _messages.IntegerField(3)
    lastModifiedAt = _messages.IntegerField(4)
    name = _messages.StringField(5)
    startTime = _messages.IntegerField(6)