from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DistanceTypeValueValuesEnum(_messages.Enum):
    """Distance type for clustering models.

    Values:
      DISTANCE_TYPE_UNSPECIFIED: Default value.
      EUCLIDEAN: Eculidean distance.
      COSINE: Cosine distance.
    """
    DISTANCE_TYPE_UNSPECIFIED = 0
    EUCLIDEAN = 1
    COSINE = 2