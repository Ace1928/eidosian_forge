from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MatcherValueValuesEnum(_messages.Enum):
    """A predefined matcher for particular cases, other than SNI selection.

    Values:
      MATCHER_UNSPECIFIED: A matcher has't been recognized.
      PRIMARY: A primary certificate that is served when SNI wasn't specified
        in the request or SNI couldn't be found in the map.
    """
    MATCHER_UNSPECIFIED = 0
    PRIMARY = 1