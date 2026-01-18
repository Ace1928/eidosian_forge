from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LookerProjectsLocationsInstancesBackupsGetRequest(_messages.Message):
    """A LookerProjectsLocationsInstancesBackupsGetRequest object.

  Fields:
    name: Required. Format: `projects/{project}/locations/{location}/instances
      /{instance}/backups/{backup}`.
  """
    name = _messages.StringField(1, required=True)