from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TrustConfig(_messages.Message):
    """Defines a trust config.

  Messages:
    LabelsValue: Set of labels associated with a TrustConfig.

  Fields:
    allowlistedCertificates: Optional. A certificate matching an allowlisted
      certificate is always considered valid as long as the certificate is
      parseable, proof of private key possession is established, and
      constraints on the certificate's SAN field are met.
    createTime: Output only. The creation timestamp of a TrustConfig.
    description: One or more paragraphs of text description of a TrustConfig.
    etag: This checksum is computed by the server based on the value of other
      fields, and may be sent on update and delete requests to ensure the
      client has an up-to-date value before proceeding.
    labels: Set of labels associated with a TrustConfig.
    name: A user-defined name of the trust config. TrustConfig names must be
      unique globally and match pattern
      `projects/*/locations/*/trustConfigs/*`.
    trustStores: Set of trust stores to perform validation against. This field
      is supported when TrustConfig is configured with Load Balancers,
      currently not supported for SPIFFE certificate validation. Only one
      TrustStore specified is currently allowed.
    updateTime: Output only. The last update timestamp of a TrustConfig.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Set of labels associated with a TrustConfig.

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
    allowlistedCertificates = _messages.MessageField('AllowlistedCertificate', 1, repeated=True)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    etag = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    name = _messages.StringField(6)
    trustStores = _messages.MessageField('TrustStore', 7, repeated=True)
    updateTime = _messages.StringField(8)