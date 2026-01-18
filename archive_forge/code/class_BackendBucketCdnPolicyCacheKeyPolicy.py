from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackendBucketCdnPolicyCacheKeyPolicy(_messages.Message):
    """Message containing what to include in the cache key for a request for
  Cloud CDN.

  Fields:
    includeHttpHeaders: Allows HTTP request headers (by name) to be used in
      the cache key.
    queryStringWhitelist: Names of query string parameters to include in cache
      keys. Default parameters are always included. '&' and '=' will be
      percent encoded and not treated as delimiters.
  """
    includeHttpHeaders = _messages.StringField(1, repeated=True)
    queryStringWhitelist = _messages.StringField(2, repeated=True)