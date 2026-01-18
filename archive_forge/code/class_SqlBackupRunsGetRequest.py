from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlBackupRunsGetRequest(_messages.Message):
    """A SqlBackupRunsGetRequest object.

  Fields:
    id: The ID of this backup run.
    instance: Cloud SQL instance ID. This does not include the project ID.
    project: Project ID of the project that contains the instance.
  """
    id = _messages.IntegerField(1, required=True)
    instance = _messages.StringField(2, required=True)
    project = _messages.StringField(3, required=True)