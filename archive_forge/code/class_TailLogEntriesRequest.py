from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TailLogEntriesRequest(_messages.Message):
    """The parameters to TailLogEntries.

  Fields:
    bufferWindow: Optional. The amount of time to buffer log entries at the
      server before being returned to prevent out of order results due to late
      arriving log entries. Valid values are between 0-60000 milliseconds.
      Defaults to 2000 milliseconds.
    filter: Optional. Only log entries that match the filter are returned. An
      empty filter matches all log entries in the resources listed in
      resource_names. Referencing a parent resource that is not listed in
      resource_names will cause the filter to return no results. The maximum
      length of a filter is 20,000 characters.
    resourceNames: Required. Name of a parent resource from which to retrieve
      log entries: projects/[PROJECT_ID] organizations/[ORGANIZATION_ID]
      billingAccounts/[BILLING_ACCOUNT_ID] folders/[FOLDER_ID]May
      alternatively be one or more views: projects/[PROJECT_ID]/locations/[LOC
      ATION_ID]/buckets/[BUCKET_ID]/views/[VIEW_ID] organizations/[ORGANIZATIO
      N_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_ID]/views/[VIEW_ID] billin
      gAccounts/[BILLING_ACCOUNT_ID]/locations/[LOCATION_ID]/buckets/[BUCKET_I
      D]/views/[VIEW_ID] folders/[FOLDER_ID]/locations/[LOCATION_ID]/buckets/[
      BUCKET_ID]/views/[VIEW_ID]
  """
    bufferWindow = _messages.StringField(1)
    filter = _messages.StringField(2)
    resourceNames = _messages.StringField(3, repeated=True)