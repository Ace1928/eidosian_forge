from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TermEndBehaviorValueValuesEnum(_messages.Enum):
    """Determine the subscription behavior when current terms end.

    Values:
      TERM_END_BEHAVIOR_UNSPECIFIED: Default term end behavior. Should not be
        used.
      RENEW_SAME_PLAN: Renews the subscription for another term.
      PERIODIC_SAME_PLAN: Renew the Term but subscription wil continue on a
        periodic basis.
      TERMINATE_AT_TERM_END: Terminates the subscription when the current term
        ends.
      RENEW_FOR_CUSTOM_TERM: Renews the subscription for a custom term.
    """
    TERM_END_BEHAVIOR_UNSPECIFIED = 0
    RENEW_SAME_PLAN = 1
    PERIODIC_SAME_PLAN = 2
    TERMINATE_AT_TERM_END = 3
    RENEW_FOR_CUSTOM_TERM = 4