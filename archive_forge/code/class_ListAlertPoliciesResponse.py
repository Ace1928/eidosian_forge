from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAlertPoliciesResponse(_messages.Message):
    """The protocol for the ListAlertPolicies response.

  Fields:
    alertPolicies: The returned alert policies.
    nextPageToken: If there might be more results than were returned, then
      this field is set to a non-empty value. To see the additional results,
      use that value as page_token in the next call to this method.
    totalSize: The total number of alert policies in all pages. This number is
      only an estimate, and may change in subsequent pages.
      https://aip.dev/158
  """
    alertPolicies = _messages.MessageField('AlertPolicy', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    totalSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)