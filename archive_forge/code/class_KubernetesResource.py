from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KubernetesResource(_messages.Message):
    """KubernetesResource contains the YAML manifests and configuration for
  Membership Kubernetes resources in the cluster. After CreateMembership or
  UpdateMembership, these resources should be re-applied in the cluster.

  Fields:
    connectResources: Output only. The Kubernetes resources for installing the
      GKE Connect agent This field is only populated in the Membership
      returned from a successful long-running operation from CreateMembership
      or UpdateMembership. It is not populated during normal GetMembership or
      ListMemberships requests. To get the resource manifest after the initial
      registration, the caller should make a UpdateMembership call with an
      empty field mask.
    membershipCrManifest: Input only. The YAML representation of the
      Membership CR. This field is ignored for GKE clusters where Hub can read
      the CR directly. Callers should provide the CR that is currently present
      in the cluster during CreateMembership or UpdateMembership, or leave
      this field empty if none exists. The CR manifest is used to validate the
      cluster has not been registered with another Membership.
    membershipResources: Output only. Additional Kubernetes resources that
      need to be applied to the cluster after Membership creation, and after
      every update. This field is only populated in the Membership returned
      from a successful long-running operation from CreateMembership or
      UpdateMembership. It is not populated during normal GetMembership or
      ListMemberships requests. To get the resource manifest after the initial
      registration, the caller should make a UpdateMembership call with an
      empty field mask.
    resourceOptions: Optional. Options for Kubernetes resource generation.
  """
    connectResources = _messages.MessageField('ResourceManifest', 1, repeated=True)
    membershipCrManifest = _messages.StringField(2)
    membershipResources = _messages.MessageField('ResourceManifest', 3, repeated=True)
    resourceOptions = _messages.MessageField('ResourceOptions', 4)