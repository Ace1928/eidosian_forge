from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DynamicSourceSplit(_messages.Message):
    """When a task splits using WorkItemStatus.dynamic_source_split, this
  message describes the two parts of the split relative to the description of
  the current task's input.

  Fields:
    primary: Primary part (continued to be processed by worker). Specified
      relative to the previously-current source. Becomes current.
    residual: Residual part (returned to the pool of work). Specified relative
      to the previously-current source.
  """
    primary = _messages.MessageField('DerivedSource', 1)
    residual = _messages.MessageField('DerivedSource', 2)