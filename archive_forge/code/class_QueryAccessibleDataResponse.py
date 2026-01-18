from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryAccessibleDataResponse(_messages.Message):
    """Response for successful QueryAccessibleData operations. This structure
  is included in the response upon operation completion.

  Fields:
    gcsUris: List of files, each of which contains a list of data_id(s) that
      are consented for a specified use in the request.
  """
    gcsUris = _messages.StringField(1, repeated=True)