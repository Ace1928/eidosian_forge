from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTasksResponse(_messages.Message):
    """ListTasksResponse is a list of Tasks resources.

  Fields:
    apiVersion: The API version for this call such as "run.googleapis.com/v1".
    items: List of Tasks.
    kind: The kind of this resource, in this case "TasksList".
    metadata: Metadata associated with this tasks list.
    unreachable: Locations that could not be reached.
  """
    apiVersion = _messages.StringField(1)
    items = _messages.MessageField('Task', 2, repeated=True)
    kind = _messages.StringField(3)
    metadata = _messages.MessageField('ListMeta', 4)
    unreachable = _messages.StringField(5, repeated=True)