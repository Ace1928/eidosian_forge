from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSaasacceleratorManagementProvidersV1SloMetadata(_messages.Message):
    """SloMetadata contains resources required for proper SLO classification of
  the instance.

  Fields:
    nodes: Optional. List of nodes. Some producers need to use per-node
      metadata to calculate SLO. This field allows such producers to publish
      per-node SLO meta data, which will be consumed by SSA Eligibility
      Exporter and published in the form of per node metric to Monarch.
    perSliEligibility: Optional. Multiple per-instance SLI eligibilities which
      apply for individual SLIs.
    tier: Name of the SLO tier the Instance belongs to. This name will be
      expected to match the tiers specified in the service SLO configuration.
      Field is mandatory and must not be empty.
  """
    nodes = _messages.MessageField('GoogleCloudSaasacceleratorManagementProvidersV1NodeSloMetadata', 1, repeated=True)
    perSliEligibility = _messages.MessageField('GoogleCloudSaasacceleratorManagementProvidersV1PerSliSloEligibility', 2)
    tier = _messages.StringField(3)