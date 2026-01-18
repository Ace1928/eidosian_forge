from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigManagementFeatureState(_messages.Message):
    """State for Anthos Config Management

  Fields:
    binauthzState: Binauthz status
    clusterName: This field is set to the `cluster_name` field of the
      Membership Spec if it is not empty. Otherwise, it is set to the
      cluster's fleet membership name.
    configSyncState: Current sync status
    hierarchyControllerState: Hierarchy Controller status
    membershipConfig: Membership configuration in the cluster. This represents
      the actual state in the cluster, while the MembershipConfig in the
      FeatureSpec represents the intended state
    operatorState: Current install status of ACM's Operator
    policyControllerState: PolicyController status
  """
    binauthzState = _messages.MessageField('BinauthzState', 1)
    clusterName = _messages.StringField(2)
    configSyncState = _messages.MessageField('ConfigSyncState', 3)
    hierarchyControllerState = _messages.MessageField('HierarchyControllerState', 4)
    membershipConfig = _messages.MessageField('MembershipConfig', 5)
    operatorState = _messages.MessageField('OperatorState', 6)
    policyControllerState = _messages.MessageField('PolicyControllerState', 7)