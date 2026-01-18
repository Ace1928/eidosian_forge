from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DnsAuthorization(_messages.Message):
    """A DnsAuthorization resource describes a way to perform domain
  authorization for certificate issuance.

  Messages:
    LabelsValue: Set of labels associated with a DnsAuthorization.

  Fields:
    createTime: Output only. The creation timestamp of a DnsAuthorization.
    description: One or more paragraphs of text description of a
      DnsAuthorization.
    dnsResourceRecord: Output only. DNS Resource Record that needs to be added
      to DNS configuration.
    domain: Required. Immutable. A domain which is being authorized. A
      DnsAuthorization resource covers a single domain and its wildcard, e.g.
      authorization for `example.com` can be used to issue certificates for
      `example.com` and `*.example.com`.
    labels: Set of labels associated with a DnsAuthorization.
    name: A user-defined name of the dns authorization. DnsAuthorization names
      must be unique globally and match pattern
      `projects/*/locations/*/dnsAuthorizations/*`.
    updateTime: Output only. The last update timestamp of a DnsAuthorization.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Set of labels associated with a DnsAuthorization.

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
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    dnsResourceRecord = _messages.MessageField('DnsResourceRecord', 3)
    domain = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    name = _messages.StringField(6)
    updateTime = _messages.StringField(7)