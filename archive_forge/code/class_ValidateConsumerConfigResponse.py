from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValidateConsumerConfigResponse(_messages.Message):
    """A ValidateConsumerConfigResponse object.

  Enums:
    ValidationErrorValueValuesEnum: The first validation which failed.

  Fields:
    existingSubnetworkCandidates: List of subnetwork candidates from the
      request which exist with the `ip_cidr_range`,
      `secondary_ip_cider_ranges`, and `outside_allocation` fields set.
    isValid: Indicates whether all the requested validations passed.
    validationError: The first validation which failed.
  """

    class ValidationErrorValueValuesEnum(_messages.Enum):
        """The first validation which failed.

    Values:
      VALIDATION_ERROR_UNSPECIFIED: <no description>
      VALIDATION_NOT_REQUESTED: In case none of the validations are requested.
      SERVICE_NETWORKING_NOT_ENABLED: <no description>
      NETWORK_NOT_FOUND: The network provided by the consumer does not exist.
      NETWORK_NOT_PEERED: The network has not been peered with the producer
        org.
      NETWORK_PEERING_DELETED: The peering was created and later deleted.
      NETWORK_NOT_IN_CONSUMERS_PROJECT: The network is a regular VPC but the
        network is not in the consumer's project.
      NETWORK_NOT_IN_CONSUMERS_HOST_PROJECT: The consumer project is a service
        project, and network is a shared VPC, but the network is not in the
        host project of this consumer project.
      HOST_PROJECT_NOT_FOUND: The host project associated with the consumer
        project was not found.
      CONSUMER_PROJECT_NOT_SERVICE_PROJECT: The consumer project is not a
        service project for the specified host project.
      RANGES_EXHAUSTED: The reserved IP ranges do not have enough space to
        create a subnet of desired size.
      RANGES_NOT_RESERVED: The IP ranges were not reserved.
      RANGES_DELETED_LATER: The IP ranges were reserved but deleted later.
      COMPUTE_API_NOT_ENABLED: The consumer project does not have the compute
        api enabled.
      USE_PERMISSION_NOT_FOUND: The consumer project does not have the
        permission from the host project.
    """
        VALIDATION_ERROR_UNSPECIFIED = 0
        VALIDATION_NOT_REQUESTED = 1
        SERVICE_NETWORKING_NOT_ENABLED = 2
        NETWORK_NOT_FOUND = 3
        NETWORK_NOT_PEERED = 4
        NETWORK_PEERING_DELETED = 5
        NETWORK_NOT_IN_CONSUMERS_PROJECT = 6
        NETWORK_NOT_IN_CONSUMERS_HOST_PROJECT = 7
        HOST_PROJECT_NOT_FOUND = 8
        CONSUMER_PROJECT_NOT_SERVICE_PROJECT = 9
        RANGES_EXHAUSTED = 10
        RANGES_NOT_RESERVED = 11
        RANGES_DELETED_LATER = 12
        COMPUTE_API_NOT_ENABLED = 13
        USE_PERMISSION_NOT_FOUND = 14
    existingSubnetworkCandidates = _messages.MessageField('Subnetwork', 1, repeated=True)
    isValid = _messages.BooleanField(2)
    validationError = _messages.EnumField('ValidationErrorValueValuesEnum', 3)