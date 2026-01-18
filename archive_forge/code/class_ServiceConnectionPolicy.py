from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceConnectionPolicy(_messages.Message):
    """The ServiceConnectionPolicy resource. Next id: 12

  Enums:
    InfrastructureValueValuesEnum: Output only. The type of underlying
      resources used to create the connection.

  Messages:
    LabelsValue: User-defined labels.

  Fields:
    createTime: Output only. Time when the ServiceConnectionMap was created.
    description: A description of this resource.
    etag: Optional. The etag is computed by the server, and may be sent on
      update and delete requests to ensure the client has an up-to-date value
      before proceeding.
    infrastructure: Output only. The type of underlying resources used to
      create the connection.
    labels: User-defined labels.
    name: Immutable. The name of a ServiceConnectionPolicy. Format: projects/{
      project}/locations/{location}/serviceConnectionPolicies/{service_connect
      ion_policy} See: https://google.aip.dev/122#fields-representing-
      resource-names
    network: The resource path of the consumer network. Example: -
      projects/{projectNumOrId}/global/networks/{resourceId}.
    pscConfig: Configuration used for Private Service Connect connections.
      Used when Infrastructure is PSC.
    pscConnections: Output only. [Output only] Information about each Private
      Service Connect connection.
    serviceClass: The service class identifier for which this
      ServiceConnectionPolicy is for. The service class identifier is a
      unique, symbolic representation of a ServiceClass. It is provided by the
      Service Producer. Google services have a prefix of gcp. For example,
      gcp-cloud-sql. 3rd party services do not. For example, test-
      service-a3dfcx.
    updateTime: Output only. Time when the ServiceConnectionMap was updated.
  """

    class InfrastructureValueValuesEnum(_messages.Enum):
        """Output only. The type of underlying resources used to create the
    connection.

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
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    etag = _messages.StringField(3)
    infrastructure = _messages.EnumField('InfrastructureValueValuesEnum', 4)
    labels = _messages.MessageField('LabelsValue', 5)
    name = _messages.StringField(6)
    network = _messages.StringField(7)
    pscConfig = _messages.MessageField('PscConfig', 8)
    pscConnections = _messages.MessageField('PscConnection', 9, repeated=True)
    serviceClass = _messages.StringField(10)
    updateTime = _messages.StringField(11)