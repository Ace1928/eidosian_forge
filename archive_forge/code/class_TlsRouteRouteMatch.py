from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TlsRouteRouteMatch(_messages.Message):
    """RouteMatch defines the predicate used to match requests to a given
  action. Multiple match types are "AND"ed for evaluation. If no routeMatch
  field is specified, this rule will unconditionally match traffic.

  Fields:
    alpn: Optional. ALPN (Application-Layer Protocol Negotiation) to match
      against. Examples: "http/1.1", "h2". At least one of sni_host and alpn
      is required. Up to 5 alpns across all matches can be set.
    sniHost: Optional. SNI (server name indicator) to match against. SNI will
      be matched against all wildcard domains, i.e. `www.example.com` will be
      first matched against `www.example.com`, then `*.example.com`, then
      `*.com.` Partial wildcards are not supported, and values like
      *w.example.com are invalid. At least one of sni_host and alpn is
      required. Up to 5 sni hosts across all matches can be set.
  """
    alpn = _messages.StringField(1, repeated=True)
    sniHost = _messages.StringField(2, repeated=True)