from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReserveIdsRequest(_messages.Message):
    """The request for Datastore.ReserveIds.

  Fields:
    databaseId: The ID of the database against which to make the request.
      '(default)' is not allowed; please use empty string '' to refer the
      default database.
    keys: Required. A list of keys with complete key paths whose numeric IDs
      should not be auto-allocated.
  """
    databaseId = _messages.StringField(1)
    keys = _messages.MessageField('Key', 2, repeated=True)