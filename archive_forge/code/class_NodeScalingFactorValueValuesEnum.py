from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeScalingFactorValueValuesEnum(_messages.Enum):
    """Immutable. The node scaling factor of this cluster.

    Values:
      NODE_SCALING_FACTOR_UNSPECIFIED: No node scaling specified. Defaults to
        NODE_SCALING_FACTOR_1X.
      NODE_SCALING_FACTOR_1X: The cluster is running with a scaling factor of
        1.
      NODE_SCALING_FACTOR_2X: The cluster is running with a scaling factor of
        2. All node count values must be in increments of 2 with this scaling
        factor enabled, otherwise an INVALID_ARGUMENT error will be returned.
    """
    NODE_SCALING_FACTOR_UNSPECIFIED = 0
    NODE_SCALING_FACTOR_1X = 1
    NODE_SCALING_FACTOR_2X = 2