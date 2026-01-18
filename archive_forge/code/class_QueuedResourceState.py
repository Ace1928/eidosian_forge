from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueuedResourceState(_messages.Message):
    """QueuedResourceState defines the details of the QueuedResource request.

  Enums:
    StateValueValuesEnum: Output only. State of the QueuedResource request.
    StateInitiatorValueValuesEnum: Output only. The initiator of the
      QueuedResources's current state. Used to indicate whether the
      SUSPENDING/SUSPENDED state was initiated by the user or the service.

  Fields:
    acceptedData: Output only. Further data for the accepted state.
    activeData: Output only. Further data for the active state.
    creatingData: Output only. Further data for the creating state.
    deletingData: Output only. Further data for the deleting state.
    failedData: Output only. Further data for the failed state.
    provisioningData: Output only. Further data for the provisioning state.
    state: Output only. State of the QueuedResource request.
    stateInitiator: Output only. The initiator of the QueuedResources's
      current state. Used to indicate whether the SUSPENDING/SUSPENDED state
      was initiated by the user or the service.
    suspendedData: Output only. Further data for the suspended state.
    suspendingData: Output only. Further data for the suspending state.
  """

    class StateInitiatorValueValuesEnum(_messages.Enum):
        """Output only. The initiator of the QueuedResources's current state.
    Used to indicate whether the SUSPENDING/SUSPENDED state was initiated by
    the user or the service.

    Values:
      STATE_INITIATOR_UNSPECIFIED: The state initiator is unspecified.
      USER: The current QueuedResource state was initiated by the user.
      SERVICE: The current QueuedResource state was initiated by the service.
    """
        STATE_INITIATOR_UNSPECIFIED = 0
        USER = 1
        SERVICE = 2

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the QueuedResource request.

    Values:
      STATE_UNSPECIFIED: State of the QueuedResource request is not known/set.
      CREATING: The QueuedResource request has been received. We're still
        working on determining if we will be able to honor this request.
      ACCEPTED: The QueuedResource request has passed initial
        validation/admission control and has been persisted in the queue.
      PROVISIONING: The QueuedResource request has been selected. The
        associated resources are currently being provisioned (or very soon
        will begin provisioning).
      FAILED: The request could not be completed. This may be due to some
        late-discovered problem with the request itself, or due to
        unavailability of resources within the constraints of the request
        (e.g., the 'valid until' start timing constraint expired).
      DELETING: The QueuedResource is being deleted.
      ACTIVE: The resources specified in the QueuedResource request have been
        provisioned and are ready for use by the end-user/consumer.
      SUSPENDING: The resources specified in the QueuedResource request are
        being deleted. This may have been initiated by the user, or the Cloud
        TPU service. Inspect the state data for more details.
      SUSPENDED: The resources specified in the QueuedResource request have
        been deleted.
      WAITING_FOR_RESOURCES: The QueuedResource request has passed initial
        validation and has been persisted in the queue. It will remain in this
        state until there are sufficient free resources to begin provisioning
        your request. Wait times will vary significantly depending on demand
        levels. When demand is high, not all requests can be immediately
        provisioned. If you need more reliable obtainability of TPUs consider
        purchasing a reservation. To put a limit on how long you are willing
        to wait, use [timing
        constraints](https://cloud.google.com/tpu/docs/queued-
        resources#request_a_queued_resource_before_a_specified_time).
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACCEPTED = 2
        PROVISIONING = 3
        FAILED = 4
        DELETING = 5
        ACTIVE = 6
        SUSPENDING = 7
        SUSPENDED = 8
        WAITING_FOR_RESOURCES = 9
    acceptedData = _messages.MessageField('AcceptedData', 1)
    activeData = _messages.MessageField('ActiveData', 2)
    creatingData = _messages.MessageField('CreatingData', 3)
    deletingData = _messages.MessageField('DeletingData', 4)
    failedData = _messages.MessageField('FailedData', 5)
    provisioningData = _messages.MessageField('ProvisioningData', 6)
    state = _messages.EnumField('StateValueValuesEnum', 7)
    stateInitiator = _messages.EnumField('StateInitiatorValueValuesEnum', 8)
    suspendedData = _messages.MessageField('SuspendedData', 9)
    suspendingData = _messages.MessageField('SuspendingData', 10)