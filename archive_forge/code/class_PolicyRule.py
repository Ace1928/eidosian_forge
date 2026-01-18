from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyRule(_messages.Message):
    """Rule to apply.

  Fields:
    restrictRollouts: Rollout restrictions.
  """
    restrictRollouts = _messages.MessageField('RestrictRollout', 1)