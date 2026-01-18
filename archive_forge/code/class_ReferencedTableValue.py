from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReferencedTableValue(_messages.Message):
    """A ReferencedTableValue object.

      Fields:
        datasetId: A string attribute.
        projectId: A string attribute.
        tableId: A string attribute.
      """
    datasetId = _messages.StringField(1)
    projectId = _messages.StringField(2)
    tableId = _messages.StringField(3)