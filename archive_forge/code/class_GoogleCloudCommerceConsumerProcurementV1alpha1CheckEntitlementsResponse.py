from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1CheckEntitlementsResponse(_messages.Message):
    """Response message for ConsumerProcurementService.CheckEntitlements.

  Fields:
    entitlementCandidates: Output only. Can-be-used Entitlement Candidates.
      Expected to contain at most one entitlement unless the product is opted
      in go/ccm-purchasing:flat-fee-multi-sub-design
    entitlements: Output only. Available Entitlements. Expected to contain at
      most one entitlement unless the product is opted in go/ccm-
      purchasing:flat-fee-multi-sub-design
  """
    entitlementCandidates = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1Entitlement', 1, repeated=True)
    entitlements = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1Entitlement', 2, repeated=True)