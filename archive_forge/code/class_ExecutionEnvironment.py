from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecutionEnvironment(_messages.Message):
    """Contains the workerpool.

  Fields:
    workerPool: Required. The workerpool used to run the PipelineRun.
  """
    workerPool = _messages.StringField(1)