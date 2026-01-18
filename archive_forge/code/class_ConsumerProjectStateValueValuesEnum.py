from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConsumerProjectStateValueValuesEnum(_messages.Enum):
    """The CCFE state of the consumer project. It is the same state that is
    communicated to the CLH during project events. Notice that this field is
    not set in the DB, it is only set in this proto when communicated to CLH
    in the side channel.

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