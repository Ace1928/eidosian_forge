from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1ListEventsResponse(_messages.Message):
    """Response to listing order events

  Fields:
    events: The list of events in this response.
    nextPageToken: The token for fetching the next page.
  """
    events = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1Event', 1, repeated=True)
    nextPageToken = _messages.StringField(2)