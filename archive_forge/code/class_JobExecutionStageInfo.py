from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobExecutionStageInfo(_messages.Message):
    """Contains information about how a particular google.dataflow.v1beta3.Step
  will be executed.

  Fields:
    stepName: The steps associated with the execution stage. Note that stages
      may have several steps, and that a given step might be run by more than
      one stage.
  """
    stepName = _messages.StringField(1, repeated=True)