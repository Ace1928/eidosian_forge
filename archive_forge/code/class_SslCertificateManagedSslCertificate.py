from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SslCertificateManagedSslCertificate(_messages.Message):
    """Configuration and status of a managed SSL certificate.

  Enums:
    StatusValueValuesEnum: [Output only] Status of the managed certificate
      resource.

  Messages:
    DomainStatusValue: [Output only] Detailed statuses of the domains
      specified for managed certificate resource.

  Fields:
    domainStatus: [Output only] Detailed statuses of the domains specified for
      managed certificate resource.
    domains: The domains for which a managed SSL certificate will be
      generated. Each Google-managed SSL certificate supports up to the
      [maximum number of domains per Google-managed SSL certificate](/load-
      balancing/docs/quotas#ssl_certificates).
    status: [Output only] Status of the managed certificate resource.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """[Output only] Status of the managed certificate resource.

    Values:
      ACTIVE: The certificate management is working, and a certificate has
        been provisioned.
      MANAGED_CERTIFICATE_STATUS_UNSPECIFIED: <no description>
      PROVISIONING: The certificate management is working. GCP will attempt to
        provision the first certificate.
      PROVISIONING_FAILED: Certificate provisioning failed due to an issue
        with the DNS or load balancing configuration. For details of which
        domain failed, consult domain_status field.
      PROVISIONING_FAILED_PERMANENTLY: Certificate provisioning failed due to
        an issue with the DNS or load balancing configuration. It won't be
        retried. To try again delete and create a new managed SslCertificate
        resource. For details of which domain failed, consult domain_status
        field.
      RENEWAL_FAILED: Renewal of the certificate has failed due to an issue
        with the DNS or load balancing configuration. The existing cert is
        still serving; however, it will expire shortly. To provision a renewed
        certificate, delete and create a new managed SslCertificate resource.
        For details on which domain failed, consult domain_status field.
    """
        ACTIVE = 0
        MANAGED_CERTIFICATE_STATUS_UNSPECIFIED = 1
        PROVISIONING = 2
        PROVISIONING_FAILED = 3
        PROVISIONING_FAILED_PERMANENTLY = 4
        RENEWAL_FAILED = 5

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
    domainStatus = _messages.MessageField('DomainStatusValue', 1)
    domains = _messages.StringField(2, repeated=True)
    status = _messages.EnumField('StatusValueValuesEnum', 3)