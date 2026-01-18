from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterIstio(_messages.Message):
    """Istio service scoped to a single Kubernetes cluster. Learn more at
  https://istio.io. Clusters running OSS Istio will have their services
  ingested as this type.

  Fields:
    clusterName: The name of the Kubernetes cluster in which this Istio
      service is defined. Corresponds to the cluster_name resource label in
      k8s_cluster resources.
    location: The location of the Kubernetes cluster in which this Istio
      service is defined. Corresponds to the location resource label in
      k8s_cluster resources.
    serviceName: The name of the Istio service underlying this service.
      Corresponds to the destination_service_name metric label in Istio
      metrics.
    serviceNamespace: The namespace of the Istio service underlying this
      service. Corresponds to the destination_service_namespace metric label
      in Istio metrics.
  """
    clusterName = _messages.StringField(1)
    location = _messages.StringField(2)
    serviceName = _messages.StringField(3)
    serviceNamespace = _messages.StringField(4)