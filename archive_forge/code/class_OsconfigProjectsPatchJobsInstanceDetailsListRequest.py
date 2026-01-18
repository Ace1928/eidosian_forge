from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigProjectsPatchJobsInstanceDetailsListRequest(_messages.Message):
    """A OsconfigProjectsPatchJobsInstanceDetailsListRequest object.

  Fields:
    filter: A filter expression that filters results listed in the response.
      This field supports filtering results by instance zone, name, state, or
      `failure_reason`.
    pageSize: The maximum number of instance details records to return.
      Default is 100.
    pageToken: A pagination token returned from a previous call that indicates
      where this listing should continue from.
    parent: Required. The parent for the instances are in the form of
      `projects/*/patchJobs/*`.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)