from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsLocationsBackupsListRequest(_messages.Message):
    """A FirestoreProjectsLocationsBackupsListRequest object.

  Fields:
    parent: Required. The location to list backups from. Format is
      `projects/{project}/locations/{location}`. Use `{location} = '-'` to
      list backups from all locations for the given project. This allows
      listing backups from a single location or from all locations.
  """
    parent = _messages.StringField(1, required=True)