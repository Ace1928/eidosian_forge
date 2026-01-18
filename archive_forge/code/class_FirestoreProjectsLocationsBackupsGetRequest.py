from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsLocationsBackupsGetRequest(_messages.Message):
    """A FirestoreProjectsLocationsBackupsGetRequest object.

  Fields:
    name: Required. Name of the backup to fetch. Format is
      `projects/{project}/locations/{location}/backups/{backup}`.
  """
    name = _messages.StringField(1, required=True)