from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryRoutinesDeleteRequest(_messages.Message):
    """A BigqueryRoutinesDeleteRequest object.

  Fields:
    datasetId: Required. Dataset ID of the routine to delete
    projectId: Required. Project ID of the routine to delete
    routineId: Required. Routine ID of the routine to delete
  """
    datasetId = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)
    routineId = _messages.StringField(3, required=True)