from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyList(_messages.Message):
    """`PolicyList` contains policy resources in the hierarchy ordered from
  leaf to root.

  Fields:
    policies: List of policy resources ordered from leaf to root.
  """
    policies = _messages.StringField(1, repeated=True)