from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GraphqlErrorExtensions(_messages.Message):
    """GraphqlErrorExtensions contains additional information of
  `GraphqlError`.

  Fields:
    file: The source file name where the error occurred. Included only for
      `UpdateSchema` and `UpdateConnector`, it corresponds to `File.path` of
      the provided `Source`.
  """
    file = _messages.StringField(1)