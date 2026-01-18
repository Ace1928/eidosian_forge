from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Advice(_messages.Message):
    """Generated advice about this change, used for providing more information
  about how a change will affect the existing service.

  Fields:
    description: Useful description for why this advice was applied and what
      actions should be taken to mitigate any implied risks.
  """
    description = _messages.StringField(1)