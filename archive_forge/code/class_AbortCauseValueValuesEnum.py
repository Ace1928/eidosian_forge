from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AbortCauseValueValuesEnum(_messages.Enum):
    """The reason probing was aborted.

    Values:
      PROBING_ABORT_CAUSE_UNSPECIFIED: No reason was specified.
      PERMISSION_DENIED: The user lacks permission to access some of the
        network resources required to run the test.
      NO_SOURCE_LOCATION: No valid source endpoint could be derived from the
        request.
    """
    PROBING_ABORT_CAUSE_UNSPECIFIED = 0
    PERMISSION_DENIED = 1
    NO_SOURCE_LOCATION = 2