from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RequestedVerifyOptionValueValuesEnum(_messages.Enum):
    """Requested verifiability options.

    Values:
      NOT_VERIFIED: Not a verifiable build (the default).
      VERIFIED: Build must be verified.
    """
    NOT_VERIFIED = 0
    VERIFIED = 1