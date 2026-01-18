from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CacheKeyPolicy(_messages.Message):
    """Message containing what to include in the cache key for a request for
  Cloud CDN.

  Fields:
    includeHost: If true, requests to different hosts will be cached
      separately.
    includeHttpHeaders: Allows HTTP request headers (by name) to be used in
      the cache key.
    includeNamedCookies: Allows HTTP cookies (by name) to be used in the cache
      key. The name=value pair will be used in the cache key Cloud CDN
      generates.
    includeProtocol: If true, http and https requests will be cached
      separately.
    includeQueryString: If true, include query string parameters in the cache
      key according to query_string_whitelist and query_string_blacklist. If
      neither is set, the entire query string will be included. If false, the
      query string will be excluded from the cache key entirely.
    queryStringBlacklist: Names of query string parameters to exclude in cache
      keys. All other parameters will be included. Either specify
      query_string_whitelist or query_string_blacklist, not both. '&' and '='
      will be percent encoded and not treated as delimiters.
    queryStringWhitelist: Names of query string parameters to include in cache
      keys. All other parameters will be excluded. Either specify
      query_string_whitelist or query_string_blacklist, not both. '&' and '='
      will be percent encoded and not treated as delimiters.
  """
    includeHost = _messages.BooleanField(1)
    includeHttpHeaders = _messages.StringField(2, repeated=True)
    includeNamedCookies = _messages.StringField(3, repeated=True)
    includeProtocol = _messages.BooleanField(4)
    includeQueryString = _messages.BooleanField(5)
    queryStringBlacklist = _messages.StringField(6, repeated=True)
    queryStringWhitelist = _messages.StringField(7, repeated=True)