from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentityEvalStateValueValuesEnum(_messages.Enum):
    """Output only. Details about the evaluation state of the identity set in
    policy.

    Values:
      IDENTITY_EVAL_STATE_UNSPECIFIED: Not used
      MATCH: The request matches the identity
      NOT_MATCH: The request doesn't match the identity
      NOT_SUPPORTED: The identity is not supported
      INFO_DENIED: The sender of the request is not allowed to verify the
        identity.
    """
    IDENTITY_EVAL_STATE_UNSPECIFIED = 0
    MATCH = 1
    NOT_MATCH = 2
    NOT_SUPPORTED = 3
    INFO_DENIED = 4