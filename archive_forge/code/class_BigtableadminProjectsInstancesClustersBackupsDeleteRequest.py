from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesClustersBackupsDeleteRequest(_messages.Message):
    """A BigtableadminProjectsInstancesClustersBackupsDeleteRequest object.

  Fields:
    name: Required. Name of the backup to delete. Values are of the form `proj
      ects/{project}/instances/{instance}/clusters/{cluster}/backups/{backup}`
      .
  """
    name = _messages.StringField(1, required=True)