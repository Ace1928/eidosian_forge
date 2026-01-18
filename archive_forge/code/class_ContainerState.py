from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerState(_messages.Message):
    """ContainerState contains the externally-visible container state that is
  used to communicate the state and reasoning for that state to the CLH. This
  data is not persisted by CCFE, but is instead derived from CCFE's internal
  representation of the container state.

  Enums:
    StateValueValuesEnum: The current state of the container. This state is
      the culmination of all of the opinions from external systems that CCFE
      knows about of the container.

  Fields:
    currentReasons: A Reasons attribute.
    previousReasons: The previous and current reasons for a container state
      will be sent for a container event. CLHs that need to know the signal
      that caused the container event to trigger (edges) as opposed to just
      knowing the state can act upon differences in the previous and current
      reasons.Reasons will be provided for every system: service management,
      data governance, abuse, and billing.If this is a CCFE-triggered event
      used for reconciliation then the current reasons will be set to their
      *_CONTROL_PLANE_SYNC state. The previous reasons will contain the last
      known set of non-unknown non-control_plane_sync reasons for the state.
    state: The current state of the container. This state is the culmination
      of all of the opinions from external systems that CCFE knows about of
      the container.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The current state of the container. This state is the culmination of
    all of the opinions from external systems that CCFE knows about of the
    container.

    Values:
      UNKNOWN_STATE: A container should never be in an unknown state. Receipt
        of a container with this state is an error.
      ON: CCFE considers the container to be serving or transitioning into
        serving.
      OFF: CCFE considers the container to be in an OFF state. This could
        occur due to various factors. The state could be triggered by Google-
        internal audits (ex. abuse suspension, billing closed) or cleanups
        trigged by compliance systems (ex. data governance hide). User-
        initiated events such as service management deactivation trigger a
        container to an OFF state.CLHs might choose to do nothing in this case
        or to turn off costly resources. CLHs need to consider the customer
        experience if an ON/OFF/ON sequence of state transitions occurs vs.
        the cost of deleting resources, keeping metadata about resources, or
        even keeping resources live for a period of time.CCFE will not send
        any new customer requests to the CLH when the container is in an OFF
        state. However, CCFE will allow all previous customer requests relayed
        to CLH to complete.
      DELETED: This state indicates that the container has been (or is being)
        completely removed. This is often due to a data governance purge
        request and therefore resources should be deleted when this state is
        reached.
    """
        UNKNOWN_STATE = 0
        ON = 1
        OFF = 2
        DELETED = 3
    currentReasons = _messages.MessageField('Reasons', 1)
    previousReasons = _messages.MessageField('Reasons', 2)
    state = _messages.EnumField('StateValueValuesEnum', 3)