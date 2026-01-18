from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SchemeValueValuesEnum(_messages.Enum):
    """Scheme override. When specified, the task URI scheme is replaced by
    the provided value (HTTP or HTTPS).

    Values:
      SCHEME_UNSPECIFIED: Scheme unspecified. Defaults to HTTPS.
      HTTP: Convert the scheme to HTTP, e.g., https://www.google.ca will
        change to http://www.google.ca.
      HTTPS: Convert the scheme to HTTPS, e.g., http://www.google.ca will
        change to https://www.google.ca.
    """
    SCHEME_UNSPECIFIED = 0
    HTTP = 1
    HTTPS = 2