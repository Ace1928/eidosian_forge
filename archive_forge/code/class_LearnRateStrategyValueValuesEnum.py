from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearnRateStrategyValueValuesEnum(_messages.Enum):
    """The strategy to determine learn rate for the current iteration.

    Values:
      LEARN_RATE_STRATEGY_UNSPECIFIED: Default value.
      LINE_SEARCH: Use line search to determine learning rate.
      CONSTANT: Use a constant learning rate.
    """
    LEARN_RATE_STRATEGY_UNSPECIFIED = 0
    LINE_SEARCH = 1
    CONSTANT = 2