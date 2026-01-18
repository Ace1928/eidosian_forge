from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResultValueValuesEnum(_messages.Enum):
    """The overall result of the test's configuration analysis.

    Values:
      RESULT_UNSPECIFIED: No result was specified.
      REACHABLE: Possible scenarios are: * The configuration analysis
        determined that a packet originating from the source is expected to
        reach the destination. * The analysis didn't complete because the user
        lacks permission for some of the resources in the trace. However, at
        the time the user's permission became insufficient, the trace had been
        successful so far.
      UNREACHABLE: A packet originating from the source is expected to be
        dropped before reaching the destination.
      AMBIGUOUS: The source and destination endpoints do not uniquely identify
        the test location in the network, and the reachability result contains
        multiple traces. For some traces, a packet could be delivered, and for
        others, it would not be. This result is also assigned to configuration
        analysis of return path if on its own it should be REACHABLE, but
        configuration analysis of forward path is AMBIGUOUS.
      UNDETERMINED: The configuration analysis did not complete. Possible
        reasons are: * A permissions error occurred--for example, the user
        might not have read permission for all of the resources named in the
        test. * An internal error occurred. * The analyzer received an invalid
        or unsupported argument or was unable to identify a known endpoint.
    """
    RESULT_UNSPECIFIED = 0
    REACHABLE = 1
    UNREACHABLE = 2
    AMBIGUOUS = 3
    UNDETERMINED = 4