from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OptimizationStrategyValueValuesEnum(_messages.Enum):
    """Optimization strategy for training linear regression models.

    Values:
      OPTIMIZATION_STRATEGY_UNSPECIFIED: Default value.
      BATCH_GRADIENT_DESCENT: Uses an iterative batch gradient descent
        algorithm.
      NORMAL_EQUATION: Uses a normal equation to solve linear regression
        problem.
    """
    OPTIMIZATION_STRATEGY_UNSPECIFIED = 0
    BATCH_GRADIENT_DESCENT = 1
    NORMAL_EQUATION = 2