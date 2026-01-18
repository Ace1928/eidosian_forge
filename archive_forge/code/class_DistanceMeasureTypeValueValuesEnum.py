from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DistanceMeasureTypeValueValuesEnum(_messages.Enum):
    """Optional. The distance measure used in nearest neighbor search.

    Values:
      DISTANCE_MEASURE_TYPE_UNSPECIFIED: Should not be set.
      SQUARED_L2_DISTANCE: Euclidean (L_2) Distance.
      COSINE_DISTANCE: Cosine Distance. Defined as 1 - cosine similarity. We
        strongly suggest using DOT_PRODUCT_DISTANCE + UNIT_L2_NORM instead of
        COSINE distance. Our algorithms have been more optimized for
        DOT_PRODUCT distance which, when combined with UNIT_L2_NORM, is
        mathematically equivalent to COSINE distance and results in the same
        ranking.
      DOT_PRODUCT_DISTANCE: Dot Product Distance. Defined as a negative of the
        dot product.
    """
    DISTANCE_MEASURE_TYPE_UNSPECIFIED = 0
    SQUARED_L2_DISTANCE = 1
    COSINE_DISTANCE = 2
    DOT_PRODUCT_DISTANCE = 3