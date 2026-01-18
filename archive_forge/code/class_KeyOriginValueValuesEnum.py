from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KeyOriginValueValuesEnum(_messages.Enum):
    """The key origin.

    Values:
      ORIGIN_UNSPECIFIED: Unspecified key origin.
      USER_PROVIDED: Key is provided by user.
      GOOGLE_PROVIDED: Key is provided by Google.
    """
    ORIGIN_UNSPECIFIED = 0
    USER_PROVIDED = 1
    GOOGLE_PROVIDED = 2