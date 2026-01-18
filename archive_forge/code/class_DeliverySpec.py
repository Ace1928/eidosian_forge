from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DeliverySpec(_messages.Message):
    """From https://github.com/knative/eventing/blob/master/pkg/apis/duck/v1/de
  livery_types.go

  Fields:
    backoffDelay: BackoffDelay is the delay before retrying. More information
      on Duration format: - https://www.iso.org/iso-8601-date-and-time-
      format.html - https://en.wikipedia.org/wiki/ISO_8601 For linear policy,
      backoff delay is the time interval between retries. For exponential
      policy , backoff delay is backoffDelay*2^. +optional BackoffDelay
      *string `json:"backoffDelay,omitempty"
    backoffPolicy: BackoffPolicy is the retry backoff policy (linear,
      exponential).
    deadLetterSink: DeadLetterSink is the sink receiving event that could not
      be sent to a destination.
    retry: Retry is the minimum number of retries the sender should attempt
      when sending an event before moving it to the dead letter sink.
  """
    backoffDelay = _messages.StringField(1)
    backoffPolicy = _messages.StringField(2)
    deadLetterSink = _messages.MessageField('Destination', 3)
    retry = _messages.IntegerField(4, variant=_messages.Variant.INT32)