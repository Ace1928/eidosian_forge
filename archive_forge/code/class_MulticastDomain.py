from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MulticastDomain(_messages.Message):
    """Multicast domain resource.

  Messages:
    LabelsValue: Optional. Labels as key-value pairs.

  Fields:
    adminNetwork: Required. The resource name of the multicast admin VPC
      network. Use the following format:
      `projects/{project}/locations/global/networks/{network}`.
    connection: Required. The VPC connection type for this multicast domain.
    createTime: Output only. [Output only] The timestamp when the multicast
      domain was created.
    description: Optional. An optional text description of the multicast
      domain.
    labels: Optional. Labels as key-value pairs.
    name: The resource name of the multicast domain. Use the following format:
      `projects/*/locations/global/multicastDomains/*`
    network: Optional. [Deprecated] Use `admin_network` instead. The resource
      name of the multicast producer VPC network. Use following format:
      `projects/{project}/locations/global/networks/{network}`.
    updateTime: Output only. [Output only] The timestamp when the multicast
      domain was most recently updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels as key-value pairs.

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
    adminNetwork = _messages.StringField(1)
    connection = _messages.MessageField('Connection', 2)
    createTime = _messages.StringField(3)
    description = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    name = _messages.StringField(6)
    network = _messages.StringField(7)
    updateTime = _messages.StringField(8)