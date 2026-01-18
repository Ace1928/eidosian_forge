from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OptionValueValuesEnum(_messages.Enum):
    """Options used for the validation check

    Values:
      OPTIONS_UNSPECIFIED: Default value. Standard preflight validation check
        will be used.
      SKIP_VALIDATION_CHECK_BLOCKING: Prevent failed preflight checks from
        failing.
      SKIP_VALIDATION_ALL: Skip all preflight check validations.
    """
    OPTIONS_UNSPECIFIED = 0
    SKIP_VALIDATION_CHECK_BLOCKING = 1
    SKIP_VALIDATION_ALL = 2