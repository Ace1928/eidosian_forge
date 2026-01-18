from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CollectdPayloadError(_messages.Message):
    """Describes the error status for payloads that were not written.

  Fields:
    error: Records the error status for the payload. If this field is present,
      the partial errors for nested values won't be populated.
    index: The zero-based index in
      CreateCollectdTimeSeriesRequest.collectd_payloads.
    valueErrors: Records the error status for values that were not written due
      to an error.Failed payloads for which nothing is written will not
      include partial value errors.
  """
    error = _messages.MessageField('Status', 1)
    index = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    valueErrors = _messages.MessageField('CollectdValueError', 3, repeated=True)