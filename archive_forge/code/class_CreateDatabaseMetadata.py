from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateDatabaseMetadata(_messages.Message):
    """Metadata type for the operation returned by CreateDatabase.

  Fields:
    database: The database being created.
  """
    database = _messages.StringField(1)