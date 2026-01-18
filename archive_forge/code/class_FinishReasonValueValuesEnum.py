from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FinishReasonValueValuesEnum(_messages.Enum):
    """Output only. The reason why the model stopped generating tokens. If
    empty, the model has not stopped generating the tokens.

    Values:
      FINISH_REASON_UNSPECIFIED: The finish reason is unspecified.
      STOP: Natural stop point of the model or provided stop sequence.
      MAX_TOKENS: The maximum number of tokens as specified in the request was
        reached.
      SAFETY: The token generation was stopped as the response was flagged for
        safety reasons. NOTE: When streaming the Candidate.content will be
        empty if content filters blocked the output.
      RECITATION: The token generation was stopped as the response was flagged
        for unauthorized citations.
      OTHER: All other reasons that stopped the token generation
      BLOCKLIST: The token generation was stopped as the response was flagged
        for the terms which are included from the terminology blocklist.
      PROHIBITED_CONTENT: The token generation was stopped as the response was
        flagged for the prohibited contents.
      SPII: The token generation was stopped as the response was flagged for
        Sensitive Personally Identifiable Information (SPII) contents.
    """
    FINISH_REASON_UNSPECIFIED = 0
    STOP = 1
    MAX_TOKENS = 2
    SAFETY = 3
    RECITATION = 4
    OTHER = 5
    BLOCKLIST = 6
    PROHIBITED_CONTENT = 7
    SPII = 8