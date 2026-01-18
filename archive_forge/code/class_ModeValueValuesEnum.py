from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ModeValueValuesEnum(_messages.Enum):
    """Optional. Function calling mode.

    Values:
      MODE_UNSPECIFIED: Unspecified function calling mode. This value should
        not be used.
      AUTO: Default model behavior, model decides to predict either a function
        call or a natural language repspose.
      ANY: Model is constrained to always predicting a function call only. If
        "allowed_function_names" are set, the predicted function call will be
        limited to any one of "allowed_function_names", else the predicted
        function call will be any one of the provided "function_declarations".
      NONE: Model will not predict any function call. Model behavior is same
        as when not passing any function declarations.
    """
    MODE_UNSPECIFIED = 0
    AUTO = 1
    ANY = 2
    NONE = 3