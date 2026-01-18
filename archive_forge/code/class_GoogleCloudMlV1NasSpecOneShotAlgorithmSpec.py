from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1NasSpecOneShotAlgorithmSpec(_messages.Message):
    """The spec of one shot Neural Architecture Search (NAS).

  Enums:
    OneShotAlgorithmValueValuesEnum: Optional. The one-shot Neural
      Architecture Search (NAS) algorithm type. Defaults to
      `ONE_SHOT_ALGORITHM_REINFORCEMENT_LEARNING`.

  Fields:
    oneShotAlgorithm: Optional. The one-shot Neural Architecture Search (NAS)
      algorithm type. Defaults to `ONE_SHOT_ALGORITHM_REINFORCEMENT_LEARNING`.
  """

    class OneShotAlgorithmValueValuesEnum(_messages.Enum):
        """Optional. The one-shot Neural Architecture Search (NAS) algorithm
    type. Defaults to `ONE_SHOT_ALGORITHM_REINFORCEMENT_LEARNING`.

    Values:
      ONE_SHOT_ALGORITHM_UNSPECIFIED: <no description>
      REINFORCEMENT_LEARNING: The Reinforcement Learning Algorithm for one-
        shot Neural Architecture Search (NAS).
    """
        ONE_SHOT_ALGORITHM_UNSPECIFIED = 0
        REINFORCEMENT_LEARNING = 1
    oneShotAlgorithm = _messages.EnumField('OneShotAlgorithmValueValuesEnum', 1)