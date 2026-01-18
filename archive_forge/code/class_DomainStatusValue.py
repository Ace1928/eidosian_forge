from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class DomainStatusValue(_messages.Message):
    """[Output only] Detailed statuses of the domains specified for managed
    certificate resource.

    Messages:
      AdditionalProperty: An additional property for a DomainStatusValue
        object.

    Fields:
      additionalProperties: Additional properties of type DomainStatusValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a DomainStatusValue object.

      Enums:
        ValueValueValuesEnum:

      Fields:
        key: Name of the additional property.
        value: A ValueValueValuesEnum attribute.
      """

        class ValueValueValuesEnum(_messages.Enum):
            """ValueValueValuesEnum enum type.

        Values:
          ACTIVE: A managed certificate can be provisioned, no issues for this
            domain.
          DOMAIN_STATUS_UNSPECIFIED: <no description>
          FAILED_CAA_CHECKING: Failed to check CAA records for the domain.
          FAILED_CAA_FORBIDDEN: Certificate issuance forbidden by an explicit
            CAA record for the domain.
          FAILED_NOT_VISIBLE: There seems to be problem with the user's DNS or
            load balancer configuration for this domain.
          FAILED_RATE_LIMITED: Reached rate-limit for certificates per top-
            level private domain.
          PROVISIONING: Certificate provisioning for this domain is under way.
            GCP will attempt to provision the first certificate.
        """
            ACTIVE = 0
            DOMAIN_STATUS_UNSPECIFIED = 1
            FAILED_CAA_CHECKING = 2
            FAILED_CAA_FORBIDDEN = 3
            FAILED_NOT_VISIBLE = 4
            FAILED_RATE_LIMITED = 5
            PROVISIONING = 6
        key = _messages.StringField(1)
        value = _messages.EnumField('ValueValueValuesEnum', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)