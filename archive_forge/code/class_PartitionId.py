from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartitionId(_messages.Message):
    """A partition ID identifies a grouping of entities. The grouping is always
  by project and namespace, however the namespace ID may be empty. A partition
  ID contains several dimensions: project ID and namespace ID. Partition
  dimensions: - May be `""`. - Must be valid UTF-8 bytes. - Must have values
  that match regex `[A-Za-z\\d\\.\\-_]{1,100}` If the value of any dimension
  matches regex `__.*__`, the partition is reserved/read-only. A
  reserved/read-only partition ID is forbidden in certain documented contexts.
  Foreign partition IDs (in which the project ID does not match the context
  project ID ) are discouraged. Reads and writes of foreign partition IDs may
  fail if the project is not in an active state.

  Fields:
    databaseId: If not empty, the ID of the database to which the entities
      belong.
    namespaceId: If not empty, the ID of the namespace to which the entities
      belong.
    projectId: The ID of the project to which the entities belong.
  """
    databaseId = _messages.StringField(1)
    namespaceId = _messages.StringField(2)
    projectId = _messages.StringField(3)