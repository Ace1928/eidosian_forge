from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlgorithmValueValuesEnum(_messages.Enum):
    """The search algorithm specified for the Study.

    Values:
      ALGORITHM_UNSPECIFIED: The default algorithm used by Vertex AI for
        [hyperparameter tuning](https://cloud.google.com/vertex-
        ai/docs/training/hyperparameter-tuning-overview) and [Vertex AI
        Vizier](https://cloud.google.com/vertex-ai/docs/vizier).
      GRID_SEARCH: Simple grid search within the feasible space. To use grid
        search, all parameters must be `INTEGER`, `CATEGORICAL`, or
        `DISCRETE`.
      RANDOM_SEARCH: Simple random search within the feasible space.
    """
    ALGORITHM_UNSPECIFIED = 0
    GRID_SEARCH = 1
    RANDOM_SEARCH = 2