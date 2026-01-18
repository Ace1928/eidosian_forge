from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MeshIstio(_messages.Message):
    """Istio service scoped to an Istio mesh. Anthos clusters running ASM <
  1.6.8 will have their services ingested as this type.

  Fields:
    meshUid: Identifier for the mesh in which this Istio service is defined.
      Corresponds to the mesh_uid metric label in Istio metrics.
    serviceName: The name of the Istio service underlying this service.
      Corresponds to the destination_service_name metric label in Istio
      metrics.
    serviceNamespace: The namespace of the Istio service underlying this
      service. Corresponds to the destination_service_namespace metric label
      in Istio metrics.
  """
    meshUid = _messages.StringField(1)
    serviceName = _messages.StringField(2)
    serviceNamespace = _messages.StringField(3)