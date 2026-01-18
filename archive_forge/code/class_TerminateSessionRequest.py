from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TerminateSessionRequest(_messages.Message):
    """A request to terminate an interactive session.

  Fields:
    requestId: Optional. A unique ID used to identify the request. If the
      service receives two TerminateSessionRequest (https://cloud.google.com/d
      ataproc/docs/reference/rpc/google.cloud.dataproc.v1#google.cloud.datapro
      c.v1.TerminateSessionRequest)s with the same ID, the second request is
      ignored.Recommendation: Set this value to a UUID
      (https://en.wikipedia.org/wiki/Universally_unique_identifier).The value
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
  """
    requestId = _messages.StringField(1)