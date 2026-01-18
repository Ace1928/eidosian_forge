from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DomainNoticesValueListEntryValuesEnum(_messages.Enum):
    """DomainNoticesValueListEntryValuesEnum enum type.

    Values:
      DOMAIN_NOTICE_UNSPECIFIED: The notice is undefined.
      HSTS_PRELOADED: Indicates that the domain is preloaded on the HTTP
        Strict Transport Security list in browsers. Serving a website on such
        domain requires an SSL certificate. For details, see [how to get an
        SSL certificate](https://support.google.com/domains/answer/7638036).
    """
    DOMAIN_NOTICE_UNSPECIFIED = 0
    HSTS_PRELOADED = 1