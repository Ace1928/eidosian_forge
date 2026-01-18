from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouteRule(_messages.Message):
    """The priority of a given route, including its match conditions and the
  actions to take on a request that matches.

  Fields:
    description: Optional. A human-readable description of the `RouteRule`.
    headerAction: Optional. The header actions, including adding and removing
      headers, for requests that match this route.
    matchRules: Required. The list of criteria for matching attributes of a
      request to this `RouteRule`. This list has `OR` semantics: the request
      matches this `RouteRule` when any of the MatchRules are satisfied.
      However, predicates within a given `MatchRule` have `AND` semantics. All
      predicates within a `MatchRule` must match for the request to match the
      rule. You can specify up to five match rules.
    origin: Optional. An alternate EdgeCacheOrigin resource that this route
      responds with when a matching response is not in the cache. The
      following are both valid paths to an `EdgeCacheOrigin` resource: *
      `projects/my-project/locations/global/edgeCacheOrigins/my-origin` * `my-
      origin` Only one of `origin` or url_redirect can be set.
    priority: Required. The priority of this route rule, where `1` is the
      highest priority. You cannot configure two or more `RouteRules` with the
      same priority. Priority for each rule must be set to a number between 1
      and 999 inclusive. Priority numbers can have gaps, which enable you to
      add or remove rules in the future without affecting the rest of the
      rules. For example, `1, 2, 3, 4, 5, 9, 12, 16` is a valid series of
      priority numbers to which you could add rules numbered from 6 to 8, 10
      to 11, and 13 to 15 in the future without any impact on existing rules.
    routeAction: Optional. In response to a matching path, the RouteAction
      performs advanced routing actions like URL rewrites, header
      transformations, and so forth prior to forwarding the request to the
      selected origin.
    urlRedirect: Optional. The URL redirect configuration for requests that
      match this route. Only one of origin or `url_redirect` can be set.
  """
    description = _messages.StringField(1)
    headerAction = _messages.MessageField('HeaderAction', 2)
    matchRules = _messages.MessageField('MatchRule', 3, repeated=True)
    origin = _messages.StringField(4)
    priority = _messages.IntegerField(5)
    routeAction = _messages.MessageField('RouteAction', 6)
    urlRedirect = _messages.MessageField('UrlRedirect', 7)