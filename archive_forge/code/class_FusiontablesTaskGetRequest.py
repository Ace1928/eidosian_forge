from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class FusiontablesTaskGetRequest(_messages.Message):
    """A FusiontablesTaskGetRequest object.

  Fields:
    tableId: Table to which the task belongs.
    taskId: A string attribute.
  """
    tableId = _messages.StringField(1, required=True)
    taskId = _messages.StringField(2, required=True)