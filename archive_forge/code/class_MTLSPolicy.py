from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MTLSPolicy(_messages.Message):
    """Specification of the MTLSPolicy.

  Enums:
    ClientValidationModeValueValuesEnum: When the client presents an invalid
      certificate or no certificate to the load balancer, the
      `client_validation_mode` specifies how the client connection is handled.
      Required if the policy is to be used with the external HTTPS load
      balancing. For Traffic Director it must be empty.
    TierValueValuesEnum: Mutual TLS tier. Allowed only if the policy is to be
      used with external HTTPS load balancers.

  Fields:
    clientValidationCa: Required if the policy is to be used with Traffic
      Director. For external HTTPS load balancers it must be empty. Defines
      the mechanism to obtain the Certificate Authority certificate to
      validate the client certificate.
    clientValidationMode: When the client presents an invalid certificate or
      no certificate to the load balancer, the `client_validation_mode`
      specifies how the client connection is handled. Required if the policy
      is to be used with the external HTTPS load balancing. For Traffic
      Director it must be empty.
    clientValidationTrustConfig: Reference to the TrustConfig from
      certificatemanager.googleapis.com namespace. If specified, the chain
      validation will be performed against certificates configured in the
      given TrustConfig. Allowed only if the policy is to be used with
      external HTTPS load balancers.
    tier: Mutual TLS tier. Allowed only if the policy is to be used with
      external HTTPS load balancers.
  """

    class ClientValidationModeValueValuesEnum(_messages.Enum):
        """When the client presents an invalid certificate or no certificate to
    the load balancer, the `client_validation_mode` specifies how the client
    connection is handled. Required if the policy is to be used with the
    external HTTPS load balancing. For Traffic Director it must be empty.

    Values:
      CLIENT_VALIDATION_MODE_UNSPECIFIED: Not allowed.
      ALLOW_INVALID_OR_MISSING_CLIENT_CERT: Allow connection even if
        certificate chain validation of the client certificate failed or no
        client certificate was presented. The proof of possession of the
        private key is always checked if client certificate was presented.
        This mode requires the backend to implement processing of data
        extracted from a client certificate to authenticate the peer, or to
        reject connections if the client certificate fingerprint is missing.
      REJECT_INVALID: Require a client certificate and allow connection to the
        backend only if validation of the client certificate passed. If set,
        requires a reference to non-empty TrustConfig specified in
        `client_validation_trust_config`.
    """
        CLIENT_VALIDATION_MODE_UNSPECIFIED = 0
        ALLOW_INVALID_OR_MISSING_CLIENT_CERT = 1
        REJECT_INVALID = 2

    class TierValueValuesEnum(_messages.Enum):
        """Mutual TLS tier. Allowed only if the policy is to be used with
    external HTTPS load balancers.

    Values:
      TIER_UNSPECIFIED: If tier is unspecified in the request, the system will
        choose a default value - `STANDARD` tier at present.
      STANDARD: Default Tier. Primarily for Software Providers (service to
        service/API communication).
      ADVANCED: Advanced Tier. For customers in strongly regulated
        environments, specifying longer keys, complex certificate chains.
    """
        TIER_UNSPECIFIED = 0
        STANDARD = 1
        ADVANCED = 2
    clientValidationCa = _messages.MessageField('ValidationCA', 1, repeated=True)
    clientValidationMode = _messages.EnumField('ClientValidationModeValueValuesEnum', 2)
    clientValidationTrustConfig = _messages.StringField(3)
    tier = _messages.EnumField('TierValueValuesEnum', 4)