from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiscoveredService(_messages.Message):
    """DiscoveredService is a network/api interface that exposes some
  functionality to clients for consumption over the network. A discovered
  service can be registered to a App Hub service.

  Fields:
    name: Identifier. The resource name of the discovered service. Format:
      "projects/{host-project-
      id}/locations/{location}/discoveredServices/{uuid}""
    serviceProperties: Output only. Properties of an underlying compute
      resource that can comprise a Service. These are immutable.
    serviceReference: Output only. Reference to an underlying networking
      resource that can comprise a Service. These are immutable.
  """
    name = _messages.StringField(1)
    serviceProperties = _messages.MessageField('ServiceProperties', 2)
    serviceReference = _messages.MessageField('ServiceReference', 3)