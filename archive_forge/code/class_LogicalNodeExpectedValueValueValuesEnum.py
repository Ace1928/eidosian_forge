from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LogicalNodeExpectedValueValueValuesEnum(_messages.Enum):
    """If it is a logical node, it will be TRUE or FALSE.

    Values:
      LOGICAL_NODE_EXPECTED_VALUE_UNSPECIFIED: Reserved
      LOGICAL_NODE_EXPECTED_VALUE_TRUE: True
      LOGICAL_NODE_EXPECTED_VALUE_FALSE: False
    """
    LOGICAL_NODE_EXPECTED_VALUE_UNSPECIFIED = 0
    LOGICAL_NODE_EXPECTED_VALUE_TRUE = 1
    LOGICAL_NODE_EXPECTED_VALUE_FALSE = 2