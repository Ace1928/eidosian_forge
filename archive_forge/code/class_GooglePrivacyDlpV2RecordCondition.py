from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2RecordCondition(_messages.Message):
    """A condition for determining whether a transformation should be applied
  to a field.

  Fields:
    expressions: An expression.
  """
    expressions = _messages.MessageField('GooglePrivacyDlpV2Expressions', 1)