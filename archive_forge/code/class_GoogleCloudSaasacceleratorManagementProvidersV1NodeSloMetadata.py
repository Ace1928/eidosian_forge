from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSaasacceleratorManagementProvidersV1NodeSloMetadata(_messages.Message):
    """Node information for custom per-node SLO implementations. SSA does not
  support per-node SLO, but producers can populate per-node information in
  SloMetadata for custom precomputations. SSA Eligibility Exporter will emit
  per-node metric based on this information.

  Fields:
    location: The location of the node, if different from instance location.
    nodeId: The id of the node. This should be equal to
      SaasInstanceNode.node_id.
    perSliEligibility: If present, this will override eligibility for the node
      coming from instance or exclusions for specified SLIs.
  """
    location = _messages.StringField(1)
    nodeId = _messages.StringField(2)
    perSliEligibility = _messages.MessageField('GoogleCloudSaasacceleratorManagementProvidersV1PerSliSloEligibility', 3)