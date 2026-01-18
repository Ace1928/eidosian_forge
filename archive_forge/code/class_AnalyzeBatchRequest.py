from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnalyzeBatchRequest(_messages.Message):
    """A request to analyze a batch workload.

  Fields:
    requestId: Optional. A unique ID used to identify the request. If the
      service receives two AnalyzeBatchRequest (http://cloud/dataproc/docs/ref
      erence/rpc/google.cloud.dataproc.v1#google.cloud.dataproc.v1.AnalyzeBatc
      hRequest)s with the same request_id, the second request is ignored and
      the Operation that corresponds to the first request created and stored
      in the backend is returned.Recommendation: Set this value to a UUID
      (https://en.wikipedia.org/wiki/Universally_unique_identifier).The value
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
  """
    requestId = _messages.StringField(1)