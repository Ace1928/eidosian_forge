from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1ListBackupsResponse(_messages.Message):
    """The response for FirestoreAdmin.ListBackups.

  Fields:
    backups: List of all backups for the project.
    unreachable: List of locations that existing backups were not able to be
      fetched from. Instead of failing the entire requests when a single
      location is unreachable, this response returns a partial result set and
      list of locations unable to be reached here. The request can be retried
      against a single location to get a concrete error.
  """
    backups = _messages.MessageField('GoogleFirestoreAdminV1Backup', 1, repeated=True)
    unreachable = _messages.StringField(2, repeated=True)