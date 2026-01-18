from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryRoutinesInsertRequest(_messages.Message):
    """A BigqueryRoutinesInsertRequest object.

  Fields:
    datasetId: Required. Dataset ID of the new routine
    projectId: Required. Project ID of the new routine
    routine: A Routine resource to be passed as the request body.
  """
    datasetId = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)
    routine = _messages.MessageField('Routine', 3)