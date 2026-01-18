from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KmeansInitializationMethodValueValuesEnum(_messages.Enum):
    """The method used to initialize the centroids for kmeans algorithm.

    Values:
      KMEANS_INITIALIZATION_METHOD_UNSPECIFIED: Unspecified initialization
        method.
      RANDOM: Initializes the centroids randomly.
      CUSTOM: Initializes the centroids using data specified in
        kmeans_initialization_column.
      KMEANS_PLUS_PLUS: Initializes with kmeans++.
    """
    KMEANS_INITIALIZATION_METHOD_UNSPECIFIED = 0
    RANDOM = 1
    CUSTOM = 2
    KMEANS_PLUS_PLUS = 3