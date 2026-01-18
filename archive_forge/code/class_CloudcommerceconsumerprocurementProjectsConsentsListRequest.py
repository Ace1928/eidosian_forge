from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudcommerceconsumerprocurementProjectsConsentsListRequest(_messages.Message):
    """A CloudcommerceconsumerprocurementProjectsConsentsListRequest object.

  Fields:
    agreement: Required. Leaving this field unset will throw an error. Valid
      format: commerceoffercatalog.googleapis.com/billingAccounts/{billing_acc
      ount}/offers/{offer_id}/agreements/{agreement_id}
    pageSize: The maximum number of results returned by this request.
    pageToken: The continuation token, which is used to page through large
      result sets. To get the next page of results, set this parameter to the
      value of `nextPageToken` from the previous response.
    parent: Required. Parent of consents. Current supported format: -
      billingAccounts/{billing_account}
  """
    agreement = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)