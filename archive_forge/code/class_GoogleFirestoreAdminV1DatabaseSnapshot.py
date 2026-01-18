from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1DatabaseSnapshot(_messages.Message):
    """A consistent snapshot of a database at a specific point in time.

  Fields:
    database: Required. A name of the form
      `projects/{project_id}/databases/{database_id}`
    snapshotTime: Required. The timestamp at which the database snapshot is
      taken. The requested timestamp must be a whole minute within the PITR
      window.
  """
    database = _messages.StringField(1)
    snapshotTime = _messages.StringField(2)