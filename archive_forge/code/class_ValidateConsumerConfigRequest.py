from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValidateConsumerConfigRequest(_messages.Message):
    """A ValidateConsumerConfigRequest object.

  Fields:
    checkServiceNetworkingUsePermission: Optional. The IAM permission check
      determines whether the consumer project has
      'servicenetworking.services.use' permission or not.
    consumerNetwork: Required. The network that the consumer is using to
      connect with services. Must be in the form of
      projects/{project}/global/networks/{network} {project} is a project
      number, as in '12345' {network} is network name.
    consumerProject: NETWORK_NOT_IN_CONSUMERS_PROJECT,
      NETWORK_NOT_IN_CONSUMERS_HOST_PROJECT, and HOST_PROJECT_NOT_FOUND are
      done when consumer_project is provided.
    rangeReservation: RANGES_EXHAUSTED, RANGES_EXHAUSTED, and
      RANGES_DELETED_LATER are done when range_reservation is provided.
    validateNetwork: The validations will be performed in the order listed in
      the ValidationError enum. The first failure will return. If a validation
      is not requested, then the next one will be performed.
      SERVICE_NETWORKING_NOT_ENABLED and NETWORK_NOT_PEERED checks are
      performed for all requests where validation is requested.
      NETWORK_NOT_FOUND and NETWORK_DISCONNECTED checks are done for requests
      that have validate_network set to true.
  """
    checkServiceNetworkingUsePermission = _messages.BooleanField(1)
    consumerNetwork = _messages.StringField(2)
    consumerProject = _messages.MessageField('ConsumerProject', 3)
    rangeReservation = _messages.MessageField('RangeReservation', 4)
    validateNetwork = _messages.BooleanField(5)