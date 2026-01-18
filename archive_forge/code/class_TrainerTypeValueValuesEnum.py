from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TrainerTypeValueValuesEnum(_messages.Enum):
    """TrainerTypeValueValuesEnum enum type.

    Values:
      TRAINER_TYPE_UNSPECIFIED: Default value.
      AUTOML_TRAINER: <no description>
      MODEL_GARDEN_TRAINER: <no description>
    """
    TRAINER_TYPE_UNSPECIFIED = 0
    AUTOML_TRAINER = 1
    MODEL_GARDEN_TRAINER = 2