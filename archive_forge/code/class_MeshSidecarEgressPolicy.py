from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MeshSidecarEgressPolicy(_messages.Message):
    """SidecarEgressPolicy defines how egress traffic is controlled under the
  entire mesh.

  Fields:
    disableTrafficToUnknownHosts: Optional. If set to true, we will not allow
      outbound passthrough traffic to unknown services. Instead, we will only
      allow egress traffic to backend services referenced by any routes.
      Default to False.
    hosts: Required. One or more service DNSName that should be exposed to the
      workload selected by the workload_context_selector. In a GSM-Istio Mesh,
      an egress host must be in the format namespace/dnsName, where namespace
      is the exposed services' namespace and The dnsName should be specified
      using FQDN format, optionally including a wildcard character in the
      left-most component.
    workloadContextSelector: Optional. Selects the workload where the egress
      policy should be applied to its targets. An EgressPolicy without a
      WorkloadContextSelector should always be applied to its targets. The
      following precedence rules are used to resolve conflict when multiple
      EgressPolicies are applicable to the same workload: 1) EgressPolicy with
      workload_context_selectors will take precedence first. 2) If there are
      EgressPolicy with workload_context_selectors matched, the behavior is
      undefined and any EgressPolicy with selector could be returned. 3) Then
      EgressPolicy without workloadSelector will take precedence.
  """
    disableTrafficToUnknownHosts = _messages.BooleanField(1)
    hosts = _messages.StringField(2, repeated=True)
    workloadContextSelector = _messages.MessageField('WorkloadContextSelector', 3)