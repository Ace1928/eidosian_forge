from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceConnectionMap(_messages.Message):
    """The ServiceConnectionMap resource. Next id: 15

  Enums:
    InfrastructureValueValuesEnum: Output only. The infrastructure used for
      connections between consumers/producers.

  Messages:
    LabelsValue: User-defined labels.

  Fields:
    consumerPscConfigs: The PSC configurations on consumer side.
    consumerPscConnections: Output only. PSC connection details on consumer
      side.
    createTime: Output only. Time when the ServiceConnectionMap was created.
    description: A description of this resource.
    etag: Optional. The etag is computed by the server, and may be sent on
      update and delete requests to ensure the client has an up-to-date value
      before proceeding.
    infrastructure: Output only. The infrastructure used for connections
      between consumers/producers.
    labels: User-defined labels.
    name: Immutable. The name of a ServiceConnectionMap. Format: projects/{pro
      ject}/locations/{location}/serviceConnectionMaps/{service_connection_map
      } See: https://google.aip.dev/122#fields-representing-resource-names
    producerPscConfigs: The PSC configurations on producer side.
    serviceClass: The service class identifier this ServiceConnectionMap is
      for. The user of ServiceConnectionMap create API needs to have
      networkconnecitivty.serviceclasses.use iam permission for the service
      class.
    serviceClassUri: Output only. The service class uri this
      ServiceConnectionMap is for.
    token: The token provided by the consumer. This token authenticates that
      the consumer can create a connecton within the specified project and
      network.
    updateTime: Output only. Time when the ServiceConnectionMap was updated.
  """

    class InfrastructureValueValuesEnum(_messages.Enum):
        """Output only. The infrastructure used for connections between
    consumers/producers.

    Values:
      INFRASTRUCTURE_UNSPECIFIED: An invalid infrastructure as the default
        case.
      PSC: Private Service Connect is used for connections.
    """
        INFRASTRUCTURE_UNSPECIFIED = 0
        PSC = 1

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """User-defined labels.

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
    consumerPscConfigs = _messages.MessageField('ConsumerPscConfig', 1, repeated=True)
    consumerPscConnections = _messages.MessageField('ConsumerPscConnection', 2, repeated=True)
    createTime = _messages.StringField(3)
    description = _messages.StringField(4)
    etag = _messages.StringField(5)
    infrastructure = _messages.EnumField('InfrastructureValueValuesEnum', 6)
    labels = _messages.MessageField('LabelsValue', 7)
    name = _messages.StringField(8)
    producerPscConfigs = _messages.MessageField('ProducerPscConfig', 9, repeated=True)
    serviceClass = _messages.StringField(10)
    serviceClassUri = _messages.StringField(11)
    token = _messages.StringField(12)
    updateTime = _messages.StringField(13)