from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigManagementMembershipSpec(_messages.Message):
    """**Anthos Config Management**: Configuration for a single cluster.
  Intended to parallel the ConfigManagement CR.

  Enums:
    ManagementValueValuesEnum: Enables automatic Feature management.

  Fields:
    binauthz: Binauthz conifguration for the cluster. Deprecated: This field
      will be ignored and should not be set.
    cluster: The user-specified cluster name used by Config Sync cluster-name-
      selector annotation or ClusterSelector, for applying configs to only a
      subset of clusters. Omit this field if the cluster's fleet membership
      name is used by Config Sync cluster-name-selector annotation or
      ClusterSelector. Set this field if a name different from the cluster's
      fleet membership name is used by Config Sync cluster-name-selector
      annotation or ClusterSelector.
    configSync: Config Sync configuration for the cluster.
    hierarchyController: Hierarchy Controller configuration for the cluster.
    management: Enables automatic Feature management.
    policyController: Policy Controller configuration for the cluster.
    version: Version of ACM installed.
  """

    class ManagementValueValuesEnum(_messages.Enum):
        """Enables automatic Feature management.

    Values:
      MANAGEMENT_UNSPECIFIED: Unspecified
      MANAGEMENT_AUTOMATIC: Google will manage the Feature for the cluster.
      MANAGEMENT_MANUAL: User will manually manage the Feature for the
        cluster.
    """
        MANAGEMENT_UNSPECIFIED = 0
        MANAGEMENT_AUTOMATIC = 1
        MANAGEMENT_MANUAL = 2
    binauthz = _messages.MessageField('ConfigManagementBinauthzConfig', 1)
    cluster = _messages.StringField(2)
    configSync = _messages.MessageField('ConfigManagementConfigSync', 3)
    hierarchyController = _messages.MessageField('ConfigManagementHierarchyControllerConfig', 4)
    management = _messages.EnumField('ManagementValueValuesEnum', 5)
    policyController = _messages.MessageField('ConfigManagementPolicyController', 6)
    version = _messages.StringField(7)