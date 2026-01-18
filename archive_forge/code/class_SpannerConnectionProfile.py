from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerConnectionProfile(_messages.Message):
    """Specifies connection parameters required specifically for Spanner
  databases.

  Fields:
    database: Required. The database in the spanner instance to connect to, in
      the form: "projects/my-project/instances/my-instance/databases/my-
      database"
  """
    database = _messages.StringField(1)