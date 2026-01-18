from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryRoutinesGetRequest(_messages.Message):
    """A BigqueryRoutinesGetRequest object.

  Fields:
    datasetId: Required. Dataset ID of the requested routine
    projectId: Required. Project ID of the requested routine
    readMask: If set, only the Routine fields in the field mask are returned
      in the response. If unset, all Routine fields are returned.
    routineId: Required. Routine ID of the requested routine
  """
    datasetId = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)
    readMask = _messages.StringField(3)
    routineId = _messages.StringField(4, required=True)