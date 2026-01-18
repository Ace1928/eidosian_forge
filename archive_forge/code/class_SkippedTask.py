from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SkippedTask(_messages.Message):
    """SkippedTask is used to describe the Tasks that were skipped due to their
  When Expressions evaluating to False.

  Fields:
    name: Name is the Pipeline Task name
    reason: Output only. Reason is the cause of the PipelineTask being
      skipped.
    whenExpressions: WhenExpressions is the list of checks guarding the
      execution of the PipelineTask
  """
    name = _messages.StringField(1)
    reason = _messages.StringField(2)
    whenExpressions = _messages.MessageField('WhenExpression', 3, repeated=True)