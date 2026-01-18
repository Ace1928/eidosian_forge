from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EndCondition(_messages.Message):
    """The condition that a Waiter resource is waiting for.

  Fields:
    cardinality: The cardinality of the `EndCondition`.
  """
    cardinality = _messages.MessageField('Cardinality', 1)