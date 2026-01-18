from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CDNPolicyCacheKeyPolicy(_messages.Message):
    """The request parameters that contribute to the cache key.

  Fields:
    excludeHost: Optional. If `true`, exclude a request's host from the cache
      key. Requests with different hosts share content in the cache. If
      `false` (the default), a request's host is included in the cache key.
      Requests with different hosts are stored independently. **Important:**
      Enable this only if the hosts share the same origin and content.
      Removing the host from the cache key might inadvertently result in
      different objects being cached than intended, depending on which route
      the first user matched.
    excludeQueryString: Optional. If `true`, exclude query string parameters
      from the cache key. If `false` (the default), include the query string
      parameters in the cache key according to included_query_parameters and
      excluded_query_parameters. If neither is set, the entire query string is
      included.
    excludedQueryParameters: Optional. The names of query string parameters to
      exclude from cache keys. All other parameters are included. Specify
      either included_query_parameters or excluded_query_parameters, not both.
      `&` and `=` are percent encoded and not treated as delimiters. You can
      exclude up to 20 query parameters. Each query parameter name must be
      between 1 and 32 characters long (inclusive).
    includeProtocol: Optional. If `true`, HTTP and HTTPS requests are cached
      separately.
    includedCookieNames: Optional. The names of cookies to include in cache
      keys. The cookie name and cookie value of each cookie named is used as
      part of the cache key. The following limitations apply: - Must be valid
      RFC 6265 "cookie-name" tokens - Are case sensitive - Cannot start with
      "Edge-Cache-" (case insensitive) Specifying several cookies or cookies
      that have a large range of values, such as per-user, dramatically
      impacts the cache hit rate, and might result in a higher eviction rate
      and reduced performance. You can specify up to three cookie names.
    includedHeaderNames: Optional. The names of HTTP request headers to
      include in cache keys. The value of the header field is used as part of
      the cache key. The following limitations apply: - Header names must be
      valid HTTP RFC 7230 header field values. - Header field names are case
      insensitive - You can specify up to five header names. - To include the
      HTTP method, use `:method` Refer to the documentation for the allowed
      list of header names. Specifying several headers or headers that have a
      large range of values, such as per-user, dramatically impacts the cache
      hit rate, and might result in a higher eviction rate and reduced
      performance.
    includedQueryParameters: Optional. The names of query string parameters to
      include in cache keys. All other parameters are excluded. Specify either
      included_query_parameters or excluded_query_parameters, not both. `&`
      and `=` are percent encoded and not treated as delimiters. You can
      include up to 20 query parameters. Each query parameter name must be
      between 1 and 32 characters long (inclusive).
  """
    excludeHost = _messages.BooleanField(1)
    excludeQueryString = _messages.BooleanField(2)
    excludedQueryParameters = _messages.StringField(3, repeated=True)
    includeProtocol = _messages.BooleanField(4)
    includedCookieNames = _messages.StringField(5, repeated=True)
    includedHeaderNames = _messages.StringField(6, repeated=True)
    includedQueryParameters = _messages.StringField(7, repeated=True)