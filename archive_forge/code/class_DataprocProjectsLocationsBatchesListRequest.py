from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsBatchesListRequest(_messages.Message):
    """A DataprocProjectsLocationsBatchesListRequest object.

  Fields:
    filter: Optional. A filter for the batches to return in the response.A
      filter is a logical expression constraining the values of various fields
      in each batch resource. Filters are case sensitive, and may contain
      multiple clauses combined with logical operators (AND/OR). Supported
      fields are batch_id, batch_uuid, state, create_time, and labels.e.g.
      state = RUNNING and create_time < "2023-01-01T00:00:00Z" filters for
      batches in state RUNNING that were created before 2023-01-01. state =
      RUNNING and labels.environment=production filters for batches in state
      in a RUNNING state that have a production environment label.See
      https://google.aip.dev/assets/misc/ebnf-filtering.txt for a detailed
      description of the filter syntax and a list of supported comparisons.
    orderBy: Optional. Field(s) on which to sort the list of batches.Currently
      the only supported sort orders are unspecified (empty) and create_time
      desc to sort by most recently created batches first.See
      https://google.aip.dev/132#ordering for more details.
    pageSize: Optional. The maximum number of batches to return in each
      response. The service may return fewer than this value. The default page
      size is 20; the maximum page size is 1000.
    pageToken: Optional. A page token received from a previous ListBatches
      call. Provide this token to retrieve the subsequent page.
    parent: Required. The parent, which owns this collection of batches.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)