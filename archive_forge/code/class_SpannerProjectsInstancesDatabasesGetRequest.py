from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesGetRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesGetRequest object.

  Fields:
    name: Required. The name of the requested database. Values are of the form
      `projects//instances//databases/`.
  """
    name = _messages.StringField(1, required=True)