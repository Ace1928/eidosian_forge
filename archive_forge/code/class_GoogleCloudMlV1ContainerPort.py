from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1ContainerPort(_messages.Message):
    """Represents a network port in a single container. This message is a
  subset of the [Kubernetes ContainerPort v1 core
  specification](https://kubernetes.io/docs/reference/generated/kubernetes-
  api/v1.18/#containerport-v1-core).

  Fields:
    containerPort: Number of the port to expose on the container. This must be
      a valid port number: 0 < PORT_NUMBER < 65536.
  """
    containerPort = _messages.IntegerField(1, variant=_messages.Variant.INT32)