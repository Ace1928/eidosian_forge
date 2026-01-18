from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KubernetesConfig(_messages.Message):
    """KubernetesConfig contains the Kubernetes runtime configuration.

  Fields:
    gatewayServiceMesh: Kubernetes Gateway API service mesh configuration.
    serviceNetworking: Kubernetes Service networking configuration.
  """
    gatewayServiceMesh = _messages.MessageField('GatewayServiceMesh', 1)
    serviceNetworking = _messages.MessageField('ServiceNetworking', 2)