from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouterConfig(_messages.Message):
    """Message for a Router resource.

  Fields:
    default_route: Deprecated. Use the DomainConfig instead. The default route
      config. The URL paths field is not required for this route config.
    dns_zone: Deprecated. Use the DomainConfig instead. DNSZone represents an
      existing DNS zone for the router. It's used for bring-your-own-DNSZone
      case. If empty, a new managed DNS zone shall be created.
    domain: Deprecated. Use the DomainConfig instead. Domain name to associate
      with the router.
    domains: The config for each domain.
    routes: Deprecated. Use the DomainConfig instead. A list of route
      configurations to associate with the router. Each Route configuration
      must include a paths configuration.
  """
    default_route = _messages.MessageField('Route', 1)
    dns_zone = _messages.StringField(2)
    domain = _messages.StringField(3)
    domains = _messages.MessageField('DomainConfig', 4, repeated=True)
    routes = _messages.MessageField('Route', 5, repeated=True)