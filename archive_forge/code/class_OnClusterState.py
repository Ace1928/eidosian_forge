from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OnClusterState(_messages.Message):
    """OnClusterState represents the state of a sub-component of Policy
  Controller.

  Enums:
    StateValueValuesEnum: The lifecycle state of this component.

  Fields:
    details: Surface potential errors or information logs.
    state: The lifecycle state of this component.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The lifecycle state of this component.

    Values:
      LIFECYCLE_STATE_UNSPECIFIED: The lifecycle state is unspecified.
      NOT_INSTALLED: Policy Controller (PC) does not exist on the given
        cluster, and no k8s resources of any type that are associated with the
        PC should exist there. The cluster does not possess a membership with
        the Hub Feature controller.
      INSTALLING: The Hub Feature controller possesses a Membership, however
        Policy Controller is not fully installed on the cluster. In this state
        the hub can be expected to be taking actions to install the PC on the
        cluster.
      ACTIVE: Policy Controller (PC) is fully installed on the cluster and in
        an operational mode. In this state the Hub Feature controller will be
        reconciling state with the PC, and the PC will be performing it's
        operational tasks per that software. Entering a READY state requires
        that the hub has confirmed the PC is installed and its pods are
        operational with the version of the PC the Hub Feature controller
        expects.
      UPDATING: Policy Controller (PC) is fully installed, but in the process
        of changing the configuration (including changing the version of PC
        either up and down, or modifying the manifests of PC) of the resources
        running on the cluster. The Hub Feature controller has a Membership,
        is aware of the version the cluster should be running in, but has not
        confirmed for itself that the PC is running with that version.
      DECOMMISSIONING: Policy Controller (PC) may have resources on the
        cluster, but the Hub Feature controller wishes to remove the
        Membership. The Membership still exists.
      CLUSTER_ERROR: Policy Controller (PC) is not operational, and the Hub
        Feature controller is unable to act to make it operational. Entering a
        CLUSTER_ERROR state happens automatically when the PCH determines that
        a PC installed on the cluster is non-operative or that the cluster
        does not meet requirements set for the Hub Feature controller to
        administer the cluster but has nevertheless been given an instruction
        to do so (such as 'install').
      HUB_ERROR: In this state, the PC may still be operational, and only the
        Hub Feature controller is unable to act. The hub should not issue
        instructions to change the PC state, or otherwise interfere with the
        on-cluster resources. Entering a HUB_ERROR state happens automatically
        when the Hub Feature controller determines the hub is in an unhealthy
        state and it wishes to 'take hands off' to avoid corrupting the PC or
        other data.
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
    details = _messages.StringField(1)
    state = _messages.EnumField('StateValueValuesEnum', 2)