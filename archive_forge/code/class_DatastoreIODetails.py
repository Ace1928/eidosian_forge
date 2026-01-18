from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastoreIODetails(_messages.Message):
    """Metadata for a Datastore connector used by the job.

  Fields:
    namespace: Namespace used in the connection.
    projectId: ProjectId accessed in the connection.
  """
    namespace = _messages.StringField(1)
    projectId = _messages.StringField(2)