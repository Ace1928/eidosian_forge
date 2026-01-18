from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LossTypeValueValuesEnum(_messages.Enum):
    """Type of loss function used during training run.

    Values:
      LOSS_TYPE_UNSPECIFIED: Default value.
      MEAN_SQUARED_LOSS: Mean squared loss, used for linear regression.
      MEAN_LOG_LOSS: Mean log loss, used for logistic regression.
    """
    LOSS_TYPE_UNSPECIFIED = 0
    MEAN_SQUARED_LOSS = 1
    MEAN_LOG_LOSS = 2