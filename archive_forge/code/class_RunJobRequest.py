from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunJobRequest(_messages.Message):
    """Request message for creating a new execution of a job.

  Fields:
    overrides: Optional. Overrides existing job configuration for one specific
      new job execution only, using the specified values to update the job
      configuration for the new execution.
  """
    overrides = _messages.MessageField('Overrides', 1)