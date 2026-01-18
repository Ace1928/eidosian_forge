from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DistanceMeasureValueValuesEnum(_messages.Enum):
    """Required. The Distance Measure to use, required.

    Values:
      DISTANCE_MEASURE_UNSPECIFIED: Should not be set.
      EUCLIDEAN: Measures the EUCLIDEAN distance between the vectors. See
        [Euclidean](https://en.wikipedia.org/wiki/Euclidean_distance) to learn
        more
      COSINE: Compares vectors based on the angle between them, which allows
        you to measure similarity that isn't based on the vectors magnitude.
        We recommend using DOT_PRODUCT with unit normalized vectors instead of
        COSINE distance, which is mathematically equivalent with better
        performance. See [Cosine
        Similarity](https://en.wikipedia.org/wiki/Cosine_similarity) to learn
        more.
      DOT_PRODUCT: Similar to cosine but is affected by the magnitude of the
        vectors. See [Dot Product](https://en.wikipedia.org/wiki/Dot_product)
        to learn more.
    """
    DISTANCE_MEASURE_UNSPECIFIED = 0
    EUCLIDEAN = 1
    COSINE = 2
    DOT_PRODUCT = 3