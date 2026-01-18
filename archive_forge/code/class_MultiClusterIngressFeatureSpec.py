from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MultiClusterIngressFeatureSpec(_messages.Message):
    """**Multi-cluster Ingress**: The configuration for the MultiClusterIngress
  feature.

  Enums:
    BillingValueValuesEnum: Deprecated: This field will be ignored and should
      not be set. Customer's billing structure.

  Fields:
    billing: Deprecated: This field will be ignored and should not be set.
      Customer's billing structure.
    configMembership: Fully-qualified Membership name which hosts the
      MultiClusterIngress CRD. Example: `projects/foo-
      proj/locations/global/memberships/bar`
  """

    class BillingValueValuesEnum(_messages.Enum):
        """Deprecated: This field will be ignored and should not be set.
    Customer's billing structure.

    Values:
      BILLING_UNSPECIFIED: Unknown
      PAY_AS_YOU_GO: User pays a fee per-endpoint.
      ANTHOS_LICENSE: User is paying for Anthos as a whole.
    """
        BILLING_UNSPECIFIED = 0
        PAY_AS_YOU_GO = 1
        ANTHOS_LICENSE = 2
    billing = _messages.EnumField('BillingValueValuesEnum', 1)
    configMembership = _messages.StringField(2)