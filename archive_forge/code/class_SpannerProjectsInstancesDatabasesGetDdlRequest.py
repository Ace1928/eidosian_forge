from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesGetDdlRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesGetDdlRequest object.

  Fields:
    database: Required. The database whose schema we wish to get. Values are
      of the form `projects//instances//databases/`
  """
    database = _messages.StringField(1, required=True)