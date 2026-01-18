from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackendBucketCdnPolicyBypassCacheOnRequestHeader(_messages.Message):
    """Bypass the cache when the specified request headers are present, e.g.
  Pragma or Authorization headers. Values are case insensitive. The presence
  of such a header overrides the cache_mode setting.

  Fields:
    headerName: The header field name to match on when bypassing cache. Values
      are case-insensitive.
  """
    headerName = _messages.StringField(1)