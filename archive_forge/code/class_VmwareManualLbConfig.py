from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareManualLbConfig(_messages.Message):
    """Represents configuration parameters for an already existing manual load
  balancer. Given the nature of manual load balancers it is expected that said
  load balancer will be fully managed by users. IMPORTANT: Please note that
  the Anthos On-Prem API will not generate or update ManualLB configurations
  it can only bind a pre-existing configuration to a new VMware user cluster.

  Fields:
    controlPlaneNodePort: NodePort for control plane service. The Kubernetes
      API server in the admin cluster is implemented as a Service of type
      NodePort (ex. 30968).
    ingressHttpNodePort: NodePort for ingress service's http. The ingress
      service in the admin cluster is implemented as a Service of type
      NodePort (ex. 32527).
    ingressHttpsNodePort: NodePort for ingress service's https. The ingress
      service in the admin cluster is implemented as a Service of type
      NodePort (ex. 30139).
    konnectivityServerNodePort: NodePort for konnectivity server service
      running as a sidecar in each kube-apiserver pod (ex. 30564).
  """
    controlPlaneNodePort = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    ingressHttpNodePort = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    ingressHttpsNodePort = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    konnectivityServerNodePort = _messages.IntegerField(4, variant=_messages.Variant.INT32)