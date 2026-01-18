from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificateAuthority(_messages.Message):
    """A CertificateAuthority represents an individual Certificate Authority. A
  CertificateAuthority can be used to create Certificates.

  Enums:
    StateValueValuesEnum: Output only. The State for this
      CertificateAuthority.
    TierValueValuesEnum: Output only. The CaPool.Tier of the CaPool that
      includes this CertificateAuthority.
    TypeValueValuesEnum: Required. Immutable. The Type of this
      CertificateAuthority.

  Messages:
    LabelsValue: Optional. Labels with user-defined metadata.

  Fields:
    accessUrls: Output only. URLs for accessing content published by this CA,
      such as the CA certificate and CRLs.
    caCertificateDescriptions: Output only. A structured description of this
      CertificateAuthority's CA certificate and its issuers. Ordered as self-
      to-root.
    config: Required. Immutable. The config used to create a self-signed X.509
      certificate or CSR.
    createTime: Output only. The time at which this CertificateAuthority was
      created.
    deleteTime: Output only. The time at which this CertificateAuthority was
      soft deleted, if it is in the DELETED state.
    expireTime: Output only. The time at which this CertificateAuthority will
      be permanently purged, if it is in the DELETED state.
    gcsBucket: Immutable. The name of a Cloud Storage bucket where this
      CertificateAuthority will publish content, such as the CA certificate
      and CRLs. This must be a bucket name, without any prefixes (such as
      `gs://`) or suffixes (such as `.googleapis.com`). For example, to use a
      bucket named `my-bucket`, you would simply specify `my-bucket`. If not
      specified, a managed bucket will be created.
    keySpec: Required. Immutable. Used when issuing certificates for this
      CertificateAuthority. If this CertificateAuthority is a self-signed
      CertificateAuthority, this key is also used to sign the self-signed CA
      certificate. Otherwise, it is used to sign a CSR.
    labels: Optional. Labels with user-defined metadata.
    lifetime: Required. Immutable. The desired lifetime of the CA certificate.
      Used to create the "not_before_time" and "not_after_time" fields inside
      an X.509 certificate.
    name: Output only. The resource name for this CertificateAuthority in the
      format `projects/*/locations/*/caPools/*/certificateAuthorities/*`.
    pemCaCertificates: Output only. This CertificateAuthority's certificate
      chain, including the current CertificateAuthority's certificate. Ordered
      such that the root issuer is the final element (consistent with RFC
      5246). For a self-signed CA, this will only list the current
      CertificateAuthority's certificate.
    state: Output only. The State for this CertificateAuthority.
    subordinateConfig: Optional. If this is a subordinate
      CertificateAuthority, this field will be set with the subordinate
      configuration, which describes its issuers. This may be updated, but
      this CertificateAuthority must continue to validate.
    tier: Output only. The CaPool.Tier of the CaPool that includes this
      CertificateAuthority.
    type: Required. Immutable. The Type of this CertificateAuthority.
    updateTime: Output only. The time at which this CertificateAuthority was
      last updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The State for this CertificateAuthority.

    Values:
      STATE_UNSPECIFIED: Not specified.
      ENABLED: Certificates can be issued from this CA. CRLs will be generated
        for this CA. The CA will be part of the CaPool's trust anchor, and
        will be used to issue certificates from the CaPool.
      DISABLED: Certificates cannot be issued from this CA. CRLs will still be
        generated. The CA will be part of the CaPool's trust anchor, but will
        not be used to issue certificates from the CaPool.
      STAGED: Certificates can be issued from this CA. CRLs will be generated
        for this CA. The CA will be part of the CaPool's trust anchor, but
        will not be used to issue certificates from the CaPool.
      AWAITING_USER_ACTIVATION: Certificates cannot be issued from this CA.
        CRLs will not be generated. The CA will not be part of the CaPool's
        trust anchor, and will not be used to issue certificates from the
        CaPool.
      DELETED: Certificates cannot be issued from this CA. CRLs will not be
        generated. The CA may still be recovered by calling
        CertificateAuthorityService.UndeleteCertificateAuthority before
        expire_time. The CA will not be part of the CaPool's trust anchor, and
        will not be used to issue certificates from the CaPool.
    """
        STATE_UNSPECIFIED = 0
        ENABLED = 1
        DISABLED = 2
        STAGED = 3
        AWAITING_USER_ACTIVATION = 4
        DELETED = 5

    class TierValueValuesEnum(_messages.Enum):
        """Output only. The CaPool.Tier of the CaPool that includes this
    CertificateAuthority.

    Values:
      TIER_UNSPECIFIED: Not specified.
      ENTERPRISE: Enterprise tier.
      DEVOPS: DevOps tier.
    """
        TIER_UNSPECIFIED = 0
        ENTERPRISE = 1
        DEVOPS = 2

    class TypeValueValuesEnum(_messages.Enum):
        """Required. Immutable. The Type of this CertificateAuthority.

    Values:
      TYPE_UNSPECIFIED: Not specified.
      SELF_SIGNED: Self-signed CA.
      SUBORDINATE: Subordinate CA. Could be issued by a Private CA
        CertificateAuthority or an unmanaged CA.
    """
        TYPE_UNSPECIFIED = 0
        SELF_SIGNED = 1
        SUBORDINATE = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels with user-defined metadata.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    accessUrls = _messages.MessageField('AccessUrls', 1)
    caCertificateDescriptions = _messages.MessageField('CertificateDescription', 2, repeated=True)
    config = _messages.MessageField('CertificateConfig', 3)
    createTime = _messages.StringField(4)
    deleteTime = _messages.StringField(5)
    expireTime = _messages.StringField(6)
    gcsBucket = _messages.StringField(7)
    keySpec = _messages.MessageField('KeyVersionSpec', 8)
    labels = _messages.MessageField('LabelsValue', 9)
    lifetime = _messages.StringField(10)
    name = _messages.StringField(11)
    pemCaCertificates = _messages.StringField(12, repeated=True)
    state = _messages.EnumField('StateValueValuesEnum', 13)
    subordinateConfig = _messages.MessageField('SubordinateConfig', 14)
    tier = _messages.EnumField('TierValueValuesEnum', 15)
    type = _messages.EnumField('TypeValueValuesEnum', 16)
    updateTime = _messages.StringField(17)