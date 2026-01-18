from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RoutineTypeValueValuesEnum(_messages.Enum):
    """The type of the routine.

    Values:
      ROUTINE_TYPE_UNSPECIFIED: Unspecified type.
      SCALAR_FUNCTION: Non-builtin permanent scalar function.
      PROCEDURE: Stored procedure.
    """
    ROUTINE_TYPE_UNSPECIFIED = 0
    SCALAR_FUNCTION = 1
    PROCEDURE = 2