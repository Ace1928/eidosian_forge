from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OperatorValueValuesEnum(_messages.Enum):
    """Optional. Defines the operation of node selection.

    Values:
      OPERATOR_UNSPECIFIED: Default value. This value is unused.
      IN: Requires Compute Engine to seek for matched nodes.
      NOT_IN: Requires Compute Engine to avoid certain nodes.
    """
    OPERATOR_UNSPECIFIED = 0
    IN = 1
    NOT_IN = 2