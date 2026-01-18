from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyControllerMembershipState(_messages.Message):
    """**Policy Controller**: State for a single cluster.

  Enums:
    StateValueValuesEnum: The overall Policy Controller lifecycle state
      observed by the Hub Feature controller.

  Messages:
    ComponentStatesValue: Currently these include (also serving as map keys):
      1. "admission" 2. "audit" 3. "mutation"

  Fields:
    componentStates: Currently these include (also serving as map keys): 1.
      "admission" 2. "audit" 3. "mutation"
    policyContentState: The overall content state observed by the Hub Feature
      controller.
    state: The overall Policy Controller lifecycle state observed by the Hub
      Feature controller.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The overall Policy Controller lifecycle state observed by the Hub
    Feature controller.

    Values:
      LIFECYCLE_STATE_UNSPECIFIED: The lifecycle state is unspecified.
      NOT_INSTALLED: The PC does not exist on the given cluster, and no k8s
        resources of any type that are associated with the PC should exist
        there. The cluster does not possess a membership with the PCH.
      INSTALLING: The PCH possesses a Membership, however the PC is not fully
        installed on the cluster. In this state the hub can be expected to be
        taking actions to install the PC on the cluster.
      ACTIVE: The PC is fully installed on the cluster and in an operational
        mode. In this state PCH will be reconciling state with the PC, and the
        PC will be performing it's operational tasks per that software.
        Entering a READY state requires that the hub has confirmed the PC is
        installed and its pods are operational with the version of the PC the
        PCH expects.
      UPDATING: The PC is fully installed, but in the process of changing the
        configuration (including changing the version of PC either up and
        down, or modifying the manifests of PC) of the resources running on
        the cluster. The PCH has a Membership, is aware of the version the
        cluster should be running in, but has not confirmed for itself that
        the PC is running with that version.
      DECOMMISSIONING: The PC may have resources on the cluster, but the PCH
        wishes to remove the Membership. The Membership still exists.
      CLUSTER_ERROR: The PC is not operational, and the PCH is unable to act
        to make it operational. Entering a CLUSTER_ERROR state happens
        automatically when the PCH determines that a PC installed on the
        cluster is non-operative or that the cluster does not meet
        requirements set for the PCH to administer the cluster but has
        nevertheless been given an instruction to do so (such as 'install').
      HUB_ERROR: In this state, the PC may still be operational, and only the
        PCH is unable to act. The hub should not issue instructions to change
        the PC state, or otherwise interfere with the on-cluster resources.
        Entering a HUB_ERROR state happens automatically when the PCH
        determines the hub is in an unhealthy state and it wishes to 'take
        hands off' to avoid corrupting the PC or other data.
      SUSPENDED: Policy Controller (PC) is installed but suspended. This means
        that the policies are not enforced, but violations are still recorded
        (through audit).
      DETACHED: PoCo Hub is not taking any action to reconcile cluster
        objects. Changes to those objects will not be overwritten by PoCo Hub.
    """
        LIFECYCLE_STATE_UNSPECIFIED = 0
        NOT_INSTALLED = 1
        INSTALLING = 2
        ACTIVE = 3
        UPDATING = 4
        DECOMMISSIONING = 5
        CLUSTER_ERROR = 6
        HUB_ERROR = 7
        SUSPENDED = 8
        DETACHED = 9

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ComponentStatesValue(_messages.Message):
        """Currently these include (also serving as map keys): 1. "admission" 2.
    "audit" 3. "mutation"

    Messages:
      AdditionalProperty: An additional property for a ComponentStatesValue
        object.

    Fields:
      additionalProperties: Additional properties of type ComponentStatesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ComponentStatesValue object.

      Fields:
        key: Name of the additional property.
        value: A PolicyControllerOnClusterState attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('PolicyControllerOnClusterState', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    componentStates = _messages.MessageField('ComponentStatesValue', 1)
    policyContentState = _messages.MessageField('PolicyControllerPolicyContentState', 2)
    state = _messages.EnumField('StateValueValuesEnum', 3)