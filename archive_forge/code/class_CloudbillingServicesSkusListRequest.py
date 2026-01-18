from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudbillingServicesSkusListRequest(_messages.Message):
    """A CloudbillingServicesSkusListRequest object.

  Fields:
    currencyCode: The ISO 4217 currency code for the pricing info in the
      response proto. Will use the conversion rate as of start_time. Optional.
      If not specified USD will be used.
    endTime: Optional exclusive end time of the time range for which the
      pricing versions will be returned. Timestamps in the future are not
      allowed. The time range has to be within a single calendar month in
      America/Los_Angeles timezone. Time range as a whole is optional. If not
      specified, the latest pricing will be returned (up to 12 hours old at
      most).
    pageSize: Requested page size. Defaults to 5000.
    pageToken: A token identifying a page of results to return. This should be
      a `next_page_token` value returned from a previous `ListSkus` call. If
      unspecified, the first page of results is returned.
    parent: Required. The name of the service. Example:
      "services/DA34-426B-A397"
    startTime: Optional inclusive start time of the time range for which the
      pricing versions will be returned. Timestamps in the future are not
      allowed. The time range has to be within a single calendar month in
      America/Los_Angeles timezone. Time range as a whole is optional. If not
      specified, the latest pricing will be returned (up to 12 hours old at
      most).
  """
    currencyCode = _messages.StringField(1)
    endTime = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
    startTime = _messages.StringField(6)