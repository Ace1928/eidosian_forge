from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuleStats(_messages.Message):
    """RuleStats stores information on actions related to a certain rule.

  Fields:
    active: The number of actions that already have the same version as the
      current rule.
    updating: The number of actions with version < `version of the current
      rule`.
  """
    active = _messages.IntegerField(1)
    updating = _messages.IntegerField(2)