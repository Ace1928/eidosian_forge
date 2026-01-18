from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1CheckTrialEarlyStoppingStateMetatdata(_messages.Message):
    """This message will be placed in the metadata field of a
  google.longrunning.Operation associated with a CheckTrialEarlyStoppingState
  request.

  Fields:
    createTime: The time at which the operation was submitted.
    study: The name of the study that the trial belongs to.
    trial: The trial name.
  """
    createTime = _messages.StringField(1)
    study = _messages.StringField(2)
    trial = _messages.StringField(3)