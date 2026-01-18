from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WafServiceValueValuesEnum(_messages.Enum):
    """Required. The WAF service that uses this key.

    Values:
      WAF_SERVICE_UNSPECIFIED: Undefined WAF
      CA: Cloud Armor
      FASTLY: Fastly
      CLOUDFLARE: Cloudflare
      AKAMAI: Akamai
    """
    WAF_SERVICE_UNSPECIFIED = 0
    CA = 1
    FASTLY = 2
    CLOUDFLARE = 3
    AKAMAI = 4