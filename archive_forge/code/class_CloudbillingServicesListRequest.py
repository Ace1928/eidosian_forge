from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudbillingServicesListRequest(_messages.Message):
    """A CloudbillingServicesListRequest object.

  Fields:
    pageSize: Requested page size. Defaults to 5000.
    pageToken: A token identifying a page of results to return. This should be
      a `next_page_token` value returned from a previous `ListServices` call.
      If unspecified, the first page of results is returned.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)