from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1RestrictAllowedResourcesRequest(_messages.Message):
    """Request for restricting list of available resources in Workload
  environment.

  Enums:
    RestrictionTypeValueValuesEnum: Required. The type of restriction for
      using gcp products in the Workload environment.

  Fields:
    restrictionType: Required. The type of restriction for using gcp products
      in the Workload environment.
  """

    class RestrictionTypeValueValuesEnum(_messages.Enum):
        """Required. The type of restriction for using gcp products in the
    Workload environment.

    Values:
      RESTRICTION_TYPE_UNSPECIFIED: Unknown restriction type.
      ALLOW_ALL_GCP_RESOURCES: Allow the use all of all gcp products,
        irrespective of the compliance posture. This effectively removes
        gcp.restrictServiceUsage OrgPolicy on the AssuredWorkloads Folder.
      ALLOW_COMPLIANT_RESOURCES: Based on Workload's compliance regime,
        allowed list changes. See - https://cloud.google.com/assured-
        workloads/docs/supported-products for the list of supported resources.
      APPEND_COMPLIANT_RESOURCES: Similar to ALLOW_COMPLIANT_RESOURCES but
        adds the list of compliant resources to the existing list of compliant
        resources. Effective org-policy of the Folder is considered to ensure
        there is no disruption to the existing customer workflows.
    """
        RESTRICTION_TYPE_UNSPECIFIED = 0
        ALLOW_ALL_GCP_RESOURCES = 1
        ALLOW_COMPLIANT_RESOURCES = 2
        APPEND_COMPLIANT_RESOURCES = 3
    restrictionType = _messages.EnumField('RestrictionTypeValueValuesEnum', 1)