from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransformationChange(_messages.Message):
    """Specifies the transformation changes that should trigger notifications.

  Fields:
    ruleId: Required. Notifies for changes to any transformer invocations
      triggered by the transformation rule.
  """
    ruleId = _messages.StringField(1)