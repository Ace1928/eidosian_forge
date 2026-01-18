from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PipelineRunResult(_messages.Message):
    """PipelineRunResult used to describe the results of a pipeline

  Fields:
    name: Output only. Name of the TaskRun
    value: Output only. Value of the result.
  """
    name = _messages.StringField(1)
    value = _messages.MessageField('ResultValue', 2)